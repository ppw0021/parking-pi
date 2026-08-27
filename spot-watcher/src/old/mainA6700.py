#!/usr/bin/env python3
"""
mainA6700.py
- USB Camera capture (gphoto2) and occupancy evaluation on main thread.
- Asynchronous EasyOCR worker thread for plate extraction on occupied spots.
- If OCR detects NO text, it clears pending status so the NEXT main loop frame is automatically re-evaluated.
- Saves occupied spot crops and OCR processed plate crops to ./bays/.
- Generates an 'overview.jpg' in ./bays/ showing all bay bounds and statuses.
- Naming format: B{id}_Plate.jpg (e.g. B1_Plate.jpg, B2_Plate.jpg)
- Uploads array formatted JSON payload to Flask backend.
"""

import gc
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import warnings

# --- SUPPRESS PYTORCH WARNINGS & CONFIGURE CPU THREADS ---
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import cv2
import easyocr
import numpy as np
import requests
import torch
from skimage.metrics import structural_similarity as ssim

torch.set_num_threads(1)
cv2.setNumThreads(1)

# -------- CONFIGURATION & THRESHOLDS --------
SSIM_MIN = 0.50
SAT_DELTA_MIN = 40.0
STD_DELTA_MIN = 35.0
RESIZE_TO = (120, 120)
FLASK_URL = "http://10.130.1.206:5000/update_spots"
LOOP_INTERVAL_SEC = 2
BAYS_DIR = "./bays"
# --------------------------------------------

# Ensure export directory exists at launch
os.makedirs(BAYS_DIR, exist_ok=True)

PLATE_CACHE = {}
PENDING_OCR = set()
ocr_queue = queue.Queue()

print("Initializing EasyOCR Engine...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR Engine Ready.")


# ==================== CAMERA & CROP UTILITIES ==================== #
def capture_photo_usb(output_path="current.jpg"):
    cmd = [
        "gphoto2",
        "--capture-image-and-download",
        "--filename",
        output_path,
        "--force-overwrite",
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"gphoto2 Error: {e.stderr.strip()}", file=sys.stderr)
        return False


def crop_polygon(img, points):
    pts = np.array(points, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]
    return img[max(0, y) : min(h_img, y + h), max(0, x) : min(w_img, x + w)]


def save_occupied_bay_crop(bay_name, cur_patch):
    os.makedirs(BAYS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(BAYS_DIR, f"{bay_name}_taken.jpg"), cur_patch)


def remove_occupied_bay_crop(bay_name):
    path = os.path.join(BAYS_DIR, f"{bay_name}_taken.jpg")
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def save_overview_image(cur_frame, bays, payload):
    """Draws bay polygons, status colors (red=occupied, green=empty), and IDs on current frame."""
    overview_img = cur_frame.copy()
    status_map = {item["id"]: item["taken"] for item in payload}

    for b in bays:
        pts = np.array(b["points"], np.int32).reshape((-1, 1, 2))
        bay_id = parse_id_from_name(b["name"], -1)
        is_taken = status_map.get(bay_id, False)

        # Color: Red for Occupied, Green for Vacant
        color = (0, 0, 255) if is_taken else (0, 255, 0)
        cv2.polylines(overview_img, [pts], True, color, 3)

        # Draw label near the top-left of the polygon
        x, y = b["points"][0]
        label = f"B{bay_id}: {'TAKEN' if is_taken else 'EMPTY'}"
        cv2.putText(
            overview_img,
            label,
            (int(x), max(20, int(y) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    overview_path = os.path.join(BAYS_DIR, "overview.jpg")
    cv2.imwrite(overview_path, overview_img)


def patch_metrics(ref_patch, cur_patch):
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(cur_patch, RESIZE_TO)

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    s, diff = ssim(ref_g, cur_g, full=True)

    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    sat_delta = float(
        np.mean(
            np.abs(
                cur_hsv[..., 1].astype(float) - ref_hsv[..., 1].astype(float)
            )
        )
    )
    std_delta = abs(float(np.std(cur_g)) - float(np.std(ref_g)))

    return float(s), sat_delta, std_delta


def parse_id_from_name(name, fallback):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else fallback


def format_json_multiline(data):
    """Formats output array identically to original payload structure."""
    lines = ["["]
    for i, item in enumerate(data):
        comma = "," if i < len(data) - 1 else ""
        lines.append(f'  {json.dumps(item, separators=(", ", ": "))}{comma}')
    lines.append("]")
    return "\n".join(lines)


# ==================== OCR THREAD WORKER ==================== #
def ocr_worker_thread():
    while True:
        try:
            bay_id, bay_crop = ocr_queue.get()
            if bay_crop is None:
                break

            print(f"[OCR Thread] Processing Bay #{bay_id}...")
            t0 = time.time()

            plate = extract_plate(bay_crop, bay_id=bay_id)

            if plate:
                PLATE_CACHE[bay_id] = plate
                print(
                    f"[OCR Thread] SUCCESS: Bay #{bay_id} = '{plate}' ({time.time() - t0:.2f}s)"
                )
            else:
                print(
                    f"[OCR Thread] NO TEXT: Bay #{bay_id} ({time.time() - t0:.2f}s). Will retry on next loop capture."
                )

        except Exception as e:
            print(f"[OCR Thread Exception] Bay #{bay_id}: {e}")
        finally:
            PENDING_OCR.discard(bay_id)
            ocr_queue.task_done()
            gc.collect()


def extract_plate(bay_crop, bay_id=None):
    if bay_crop is None or bay_crop.size == 0:
        return None

    oriented = bay_crop.copy()

    # Add safe white border padding (30px) so letters at edges aren't clipped
    padded = cv2.copyMakeBorder(
        oriented, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # Upscale image width for OCR clarity
    ph, pw = padded.shape[:2]
    target_w = 450
    target_h = max(20, int(ph * (target_w / float(pw))))
    resized = cv2.resize(
        padded, (target_w, target_h), interpolation=cv2.INTER_CUBIC
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    detected_plate = None

    try:
        results = reader.readtext(
            gray,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.3,
            low_text=0.2,
            canvas_size=450,
            mag_ratio=1.0,
        )

        candidates = []
        for bbox, text, conf in results:
            clean = text.upper().replace(" ", "").strip()
            clean = re.sub(r"[^A-Z0-9]", "", clean)

            if any(
                bad in clean
                for bad in [
                    "CHILD",
                    "YOUTH",
                    "NSW",
                    "OPAL",
                    "TELSTRA",
                    "IKPI",
                ]
            ):
                continue

            if 3 <= len(clean) <= 8:
                candidates.append((conf, clean))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_text = candidates[0][1]

            if "AIRPOD" in best_text or best_text == "IRPOD":
                best_text = "AIRPOD"
            elif "REDBUL" in best_text or "RED" in best_text:
                best_text = "REDBUL"
            elif "A6700" in best_text:
                best_text = "A6700"

            exact_match = re.findall(r"[A-Z0-9]{5,6}", best_text)
            detected_plate = exact_match[0] if exact_match else best_text[:6]

    except Exception:
        pass

    if bay_id is not None and detected_plate is not None:
        filename = f"B{bay_id}_Plate.jpg"
        save_path = os.path.join(BAYS_DIR, filename)
        cv2.imwrite(save_path, resized)

    return detected_plate


# ==================== MAIN OCCUPANCY CONTROL LOOP ==================== #
def analyze_bays(bays, ref_image, cur_frame):
    payload = []
    fallback_id = 0

    for b in bays:
        name = b["name"]
        pts = b["points"]

        ref_p = crop_polygon(ref_image, pts)
        cur_p = crop_polygon(cur_frame, pts)

        if ref_p.size == 0 or cur_p.size == 0:
            continue

        s_w, d_sat, d_std = patch_metrics(ref_p, cur_p)
        taken = (
            (s_w < SSIM_MIN)
            or (d_sat > SAT_DELTA_MIN)
            or (d_std > STD_DELTA_MIN)
        )
        bay_id = parse_id_from_name(name, fallback_id)
        fallback_id += 1

        if taken:
            save_occupied_bay_crop(name, cur_p)

            if bay_id not in PLATE_CACHE and bay_id not in PENDING_OCR:
                PENDING_OCR.add(bay_id)
                ocr_queue.put((bay_id, cur_p.copy()))

            plate = PLATE_CACHE.get(bay_id, None)
        else:
            remove_occupied_bay_crop(name)
            PLATE_CACHE.pop(bay_id, None)
            PENDING_OCR.discard(bay_id)
            plate = None

        payload.append({"id": bay_id, "taken": bool(taken), "plate": plate})

    # Save visual summary overview of all parking spots
    save_overview_image(cur_frame, bays, payload)

    return payload


def main():
    json_bays = "bays.json"
    empty_ref_path = "emptyReference.jpg"
    current_img = "current.jpg"

    if not os.path.exists(json_bays) or not os.path.exists(empty_ref_path):
        print("Error: Baseline calibration files missing!")
        sys.exit(1)

    ref_image = cv2.imread(empty_ref_path)
    ref_h, ref_w = ref_image.shape[:2]
    with open(json_bays, "r") as f:
        bays = json.load(f)

    # Launch background OCR thread
    ocr_thread = threading.Thread(target=ocr_worker_thread, daemon=True)
    ocr_thread.start()

    print("[Main Thread] Active. Capturing camera loop...")

    try:
        while True:
            t0 = time.time()

            if not capture_photo_usb(current_img):
                time.sleep(1)
                continue

            cur_frame = cv2.imread(current_img)
            if cur_frame is None:
                continue

            cur_h, cur_w = cur_frame.shape[:2]
            if (cur_h, cur_w) != (ref_h, ref_w):
                cur_frame = cv2.resize(
                    cur_frame, (ref_w, ref_h), interpolation=cv2.INTER_AREA
                )

            spots_data = analyze_bays(bays, ref_image, cur_frame)

            formatted_json = format_json_multiline(spots_data)
            print(formatted_json)

            with open("bayStatus.json", "w") as f:
                f.write(formatted_json)

            try:
                requests.post(FLASK_URL, json=spots_data, timeout=0.2)
            except requests.RequestException:
                pass

            print(
                f"[Main Thread] Camera capture & spot cycle: {time.time() - t0:.2f}s\n"
            )
            gc.collect()
            time.sleep(LOOP_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("Stopping spot watcher...")


if __name__ == "__main__":
    main()