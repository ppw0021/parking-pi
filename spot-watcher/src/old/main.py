#!/usr/bin/env python3
"""
main.py
- USB Camera capture (Bonelk ELK-63022-R) with lens undistortion & 1440x1080 cropping.
- Occupancy evaluation on main thread.
- Asynchronous EasyOCR worker thread for plate extraction on occupied spots.
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
SSIM_MIN = 0.65         # Raised: SSIM < 0.65 flags structural changes
MEAN_DELTA_MIN = 30.0   # NEW: Detects white car on dark mat or dark car on light surface
SAT_DELTA_MIN = 25.0    # Lowered to capture subtle color shifts
STD_DELTA_MIN = 20.0    # Lowered
RESIZE_TO = (120, 120)
FLASK_URL = "http://10.130.1.206:5000/update_spots"
LOOP_INTERVAL_SEC = 2
BAYS_DIR = "./bays"
CAMERA_INDEX = 0
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
def undistort_frame(frame, k1=-0.12, k2=0.0):
    """Applies radial lens distortion correction matching initialization setup."""
    h, w = frame.shape[:2]
    fx, fy = w * 0.8, h * 0.8
    cx, cy = w / 2.0, h / 2.0

    camera_matrix = np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32
    )
    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float32)

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    return cv2.undistort(
        frame, camera_matrix, dist_coeffs, None, new_camera_matrix
    )


def capture_photo_webcam(camera_index=CAMERA_INDEX):
    """Captures 1080p frame from Bonelk webcam, undistorts, and center-crops to 1440x1080."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(
            f"Error: Could not open camera at index {camera_index}.",
            file=sys.stderr,
        )
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Flush buffered frames
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("Error: Failed to capture frame from webcam.", file=sys.stderr)
        return None

    corrected = undistort_frame(frame)

    # Center-crop 1920x1080 to 1440x1080 (cuts 240px off left & right)
    start_x = (1920 - 1440) // 2
    end_x = start_x + 1440
    return corrected[:, start_x:end_x]

def crop_polygon(img, points, pad_px=15):
    """Crops the bounding box of polygon points with extra padding so edges aren't clipped."""
    pts = np.array(points, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]

    # Expand bounding box safely within image limits (x=width, y=height)
    x1 = max(0, x - pad_px)
    y1 = max(0, y - pad_px)
    x2 = min(w_img, x + w + pad_px)
    y2 = min(h_img, y + h + pad_px)

    return img[y1:y2, x1:x2]

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

        color = (0, 0, 255) if is_taken else (0, 255, 0)
        cv2.polylines(overview_img, [pts], True, color, 3)

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

    # 1. Structural Similarity
    s, _ = ssim(ref_g, cur_g, full=True)

    # 2. Mean Brightness Delta (Key trigger for white/light model cars)
    mean_delta = abs(float(np.mean(cur_g)) - float(np.mean(ref_g)))

    # 3. Saturation Delta
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    sat_delta = float(
        np.mean(
            np.abs(
                cur_hsv[..., 1].astype(float) - ref_hsv[..., 1].astype(float)
            )
        )
    )

    # 4. Standard Deviation Delta
    std_delta = abs(float(np.std(cur_g)) - float(np.std(ref_g)))

    return float(s), mean_delta, sat_delta, std_delta

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

    # 1. High-Resolution Upscaling
    h, w = bay_crop.shape[:2]
    scale_factor = max(3.5, 900.0 / float(w))
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    upscaled = cv2.resize(
        bay_crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC
    )

    # 2. Add White Margin Padding
    padded = cv2.copyMakeBorder(
        upscaled, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=[255, 255, 255]
    )

    # 3. Grayscale, Denoising & Local Contrast Enhancement
    gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    detected_plate = None

    try:
        # Run EasyOCR tuned for 1 to 6 character alphanumeric roof plates
        results = reader.readtext(
            enhanced,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1,
            paragraph=False,
            min_size=10,
            text_threshold=0.15,
            low_text=0.05,
            link_threshold=0.4,
            canvas_size=1000,
            mag_ratio=1.5,
        )

        if results:
            # Sort detected boxes Left-to-Right
            results.sort(key=lambda item: item[0][0][0])

            candidates = []
            for bbox, text, conf in results:
                # Clean non-alphanumeric characters and force uppercase
                clean = re.sub(r"[^A-Z0-9]", "", text.upper().strip())

                # Keep detections up to 6 characters
                if 1 <= len(clean) <= 6:
                    candidates.append((conf, clean))

            if candidates:
                # Pick highest confidence character match
                candidates.sort(key=lambda x: x[0], reverse=True)
                detected_plate = candidates[0][1][:6]

    except Exception as e:
        print(f"[OCR Exception] {e}")

    # Save output image sent to OCR for visual verification
    if bay_id is not None and detected_plate is not None:
        filename = f"B{bay_id}_Plate.jpg"
        save_path = os.path.join(BAYS_DIR, filename)
        cv2.imwrite(save_path, enhanced)

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

        # Unpack all 4 return values from patch_metrics
        s_w, d_mean, d_sat, d_std = patch_metrics(ref_p, cur_p)

        # Flag occupied if ANY of the threshold conditions are met
        taken = (
            (s_w < SSIM_MIN)
            or (d_mean > MEAN_DELTA_MIN)
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

    save_overview_image(cur_frame, bays, payload)
    return payload

def main():
    json_bays = "bays.json"
    empty_ref_path = "emptyReference.jpg"

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

    print("[Main Thread] Active. Capturing webcam loop...")

    try:
        while True:
            t0 = time.time()

            cur_frame = capture_photo_webcam(CAMERA_INDEX)
            if cur_frame is None:
                time.sleep(1)
                continue

            # Ensure matching dimensions with baseline reference
            cur_h, cur_w = cur_frame.shape[:2]
            if (cur_h, cur_w) != (ref_h, ref_w):
                cur_frame = cv2.resize(
                    cur_frame, (ref_w, ref_h), interpolation=cv2.INTER_AREA
                )

            # Save latest frame snapshot
            cv2.imwrite("current.jpg", cur_frame)

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