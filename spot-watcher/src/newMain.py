#!/usr/bin/env python3
import gc
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim

import lensCorrection as lens_correction  

# -------- CONFIGURATION & THRESHOLDS --------
SSIM_MIN = 0.65 
RESIZE_TO = (120, 120)

FLASK_URL = "http://10.130.1.228:5000/update_spots"
LOOP_INTERVAL_SEC = 1
BAYS_DIR = "./bays"
CAMERA_INDEX = 0
BAYS_JSON_PATH = "bays.json"
# --------------------------------------------

os.makedirs(BAYS_DIR, exist_ok=True)

# ==================== CAMERA & UTILS ==================== #
def setup_camera(camera_index=CAMERA_INDEX):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_index}")
        sys.exit(1)

    global digital_gain # Make it accessible to the capture function
    
    try:
        with open("lens_config.json", "r") as f:
            cam_conf = json.load(f)
            exposure_val = cam_conf.get("exposure", 150)
            digital_gain = cam_conf.get("digital_gain", 1.0)
    except Exception:
        exposure_val = 150
        digital_gain = 1.0

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, max(0, exposure_val))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Force the camera to only keep the 1 most recent frame in memory so it doesn't lag
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) 
    
    return cap

def capture_live_frame(cap):
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    if not ret or frame is None:
        return None

    # NEW: Apply the software sunglasses
    frame = cv2.convertScaleAbs(frame, alpha=digital_gain, beta=0)

    # Apply Lens Correction
    corrected = lens_correction.undistort_frame(frame)
    
    # Crop to 1440x1080
    start_x = (1920 - 1440) // 2
    cropped_frame = corrected[:, start_x : start_x + 1440]
    
    # Apply CLAHE Lighting Balance
    lab = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced_l = clahe.apply(l_channel)
    merged_lab = cv2.merge((balanced_l, a_channel, b_channel))
    
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def crop_polygon(img, points, pad_px=15):
    pts = np.array(points, np.int32)
    x, y, w, h = cv2.boundingRect(pts)
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x - pad_px), max(0, y - pad_px)
    x2, y2 = min(w_img, x + w + pad_px), min(h_img, y + h + pad_px)
    return img[y1:y2, x1:x2]


def parse_id_from_name(name, fallback):
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else fallback


def patch_metrics(ref_patch, cur_patch):
    """Stripped down to ONLY calculate SSIM, skipping heavy color math."""
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(cur_patch, RESIZE_TO)
    
    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)

    s, _ = ssim(ref_g, cur_g, full=True)
    return float(s)


# ==================== MAIN ANALYSIS ==================== #
def analyze_bays(bays, cur_frame):
    payload = []

    for b in bays:
        name = b["name"]
        pts = b["points"]
        bay_id = parse_id_from_name(name, 0)

        ref_path = os.path.join(BAYS_DIR, f"B{bay_id}_ref.jpg")
        if not os.path.exists(ref_path):
            ref_path = "emptyReference.jpg" 

        ref_image = cv2.imread(ref_path)
        if ref_image is None:
            continue

        ref_p = crop_polygon(ref_image, pts)
        cur_p = crop_polygon(cur_frame, pts)

        if ref_p.size == 0 or cur_p.size == 0:
            continue

        s_w = patch_metrics(ref_p, cur_p)
        taken = (s_w < SSIM_MIN)

        taken_path = os.path.join(BAYS_DIR, f"B{bay_id}_taken.jpg")

        if taken:
            cv2.imwrite(taken_path, cur_p)
        else:
            if os.path.exists(taken_path):
                try:
                    os.remove(taken_path)
                except OSError:
                    pass

        payload.append({"id": bay_id, "taken": bool(taken)})

    return payload


def main():
    if not os.path.exists(BAYS_JSON_PATH):
        print(f"Error: Calibration file '{BAYS_JSON_PATH}' missing!")
        sys.exit(1)

    with open(BAYS_JSON_PATH, "r") as f:
        bays = json.load(f)

    print("[Main Process] Starting up...")
    print(f"[Main Process] Target Flask server: {FLASK_URL}")
    print(f"[Main Process] Loop interval: {LOOP_INTERVAL_SEC}s\n")

    # Start the camera ONCE before the loop
    cap = setup_camera(CAMERA_INDEX)

    while True:
        try:
            t0 = time.time()

            # Pull a frame instantly without restarting the hardware
            cur_frame = capture_live_frame(cap)
            if cur_frame is None:
                print("[Main Process] Frame capture failed. Retrying...")
                time.sleep(0.5)
                continue

            cv2.imwrite(os.path.join(BAYS_DIR, "currentView.jpg"), cur_frame)
            spots_data = analyze_bays(bays, cur_frame)

            status_frame = cur_frame.copy()
            status_dict = {spot["id"]: spot["taken"] for spot in spots_data}
            
            for b in bays:
                bay_id = parse_id_from_name(b["name"], 0)
                pts = np.array(b["points"], np.int32)
                is_taken = status_dict.get(bay_id, False)
                box_color = (0, 0, 255) if is_taken else (0, 255, 0)
                
                cv2.polylines(status_frame, [pts], True, box_color, 2)
                x, y = pts[0]
                cv2.putText(status_frame, b["name"], (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

            cv2.imwrite(os.path.join(BAYS_DIR, "currentBayStatus.jpg"), status_frame)

            formatted_json = (
                "[\n"
                + ",\n".join(
                    f"  {json.dumps(item, separators=(', ', ': '))}"
                    for item in spots_data
                )
                + "\n]"
            )
            print(formatted_json)

            with open("bayStatus.json", "w") as f:
                f.write(formatted_json)

            try:
                requests.post(FLASK_URL, json=spots_data, timeout=0.5)
            except Exception:
                pass

            print(f"[Main Process] Cycle completed in {time.time() - t0:.2f}s\n")
            gc.collect()
            
            time.sleep(LOOP_INTERVAL_SEC)

        except KeyboardInterrupt:
            print("\nStopping main process...")
            break
        except Exception as err:
            print(f"[Main Process Error] {err}")
            time.sleep(1)

    # Cleanly turn off the camera when we quit
    cap.release()

if __name__ == "__main__":
    main()