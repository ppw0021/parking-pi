#!/usr/bin/env python3
"""
Development version of bay availability sender.
Captures image (or uses demo file), detects bays, prints JSON status, and saves annotated bayStatus.jpg.
"""

import json, re, cv2, numpy as np, time
from skimage.metrics import structural_similarity as ssim
import requests

# -------- thresholds --------
SSIM_MIN = 0.85
SAT_DELTA_MIN = 50.0
LUMA_DELTA_MIN = 50.0
CENTER_SIGMA_FRAC = 0.25
RESIZE_TO = (120, 120)
# ----------------------------

def load_bays(path):
    with open(path, "r") as f:
        return json.load(f)

def crop_polygon(img, points):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    roi = cv2.bitwise_and(img, img, mask=mask)[y:y+h, x:x+w]
    return roi

def center_weight(h, w, sigma_frac=CENTER_SIGMA_FRAC):
    y, x = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    sigma = min(w, h) * sigma_frac
    wgt = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma * sigma))
    return wgt / (wgt.mean() + 1e-9)

def patch_metrics(ref_patch, cur_patch):
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(cur_patch, RESIZE_TO)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    ref_v, ref_s = ref_hsv[...,2].astype(np.float32), ref_hsv[...,1].astype(np.float32)
    cur_v, cur_s = cur_hsv[...,2].astype(np.float32), cur_hsv[...,1].astype(np.float32)

    H, W = RESIZE_TO[1], RESIZE_TO[0]
    WGT = center_weight(H, W)

    ref_g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    cur_g = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    s, diff = ssim(ref_g, cur_g, full=True)
    diff = 1.0 - diff
    s_weighted = 1.0 - float((diff * WGT).mean())

    sat_delta = float(((cur_s - ref_s) * WGT).mean())
    luma_delta = float(((cur_v - ref_v) * WGT).mean())

    return s_weighted, abs(sat_delta), luma_delta

def parse_id_from_name(name, fallback):
    m = re.search(r'(\d+)$', name)
    return int(m.group(1)) if m else fallback

def capture_image(save_path="current.jpg"):
    """Capture a frame from webcam and save it."""
    print("Capturing image from webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not access webcam")

    # Warm-up a few frames
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Failed to capture image")

    cv2.imwrite(save_path, frame)
    print(f"Saved {save_path}")

def analyze_and_prepare_payload(bays_path, ref_path, img_path):
    bays = load_bays(bays_path)
    ref = cv2.imread(ref_path)
    cur = cv2.imread(img_path)
    if ref is None or cur is None:
        raise SystemExit("Could not read reference or current image")

    temp = {}
    for b in bays:
        pts = b["points"]
        ref_p, cur_p = crop_polygon(ref, pts), crop_polygon(cur, pts)
        if ref_p.size == 0 or cur_p.size == 0:
            continue
        s_w, d_sat, d_luma = patch_metrics(ref_p, cur_p)
        temp[b["name"]] = (s_w, d_sat, d_luma)

    payload = []
    fallback_id = 0
    annotated = cur.copy()

    for b in bays:
        name = b["name"]
        if name not in temp:
            continue

        s_w, d_sat, d_luma = temp[name]  # raw per-spot values only
        adj_luma = abs(d_luma)           # no global compensation

        taken = (
            (s_w < SSIM_MIN) or
            (d_sat > SAT_DELTA_MIN) or
            (adj_luma > LUMA_DELTA_MIN)
        )

        bay_id = parse_id_from_name(name, fallback_id)
        fallback_id += 1
        payload.append({"id": bay_id, "taken": bool(taken)})

        color = (0, 0, 255) if taken else (0, 255, 0)
        pts = np.array(b["points"], np.int32)
        cv2.polylines(annotated, [pts], True, color, 2)
        p = pts[0]
        cv2.putText(annotated, f"{bay_id}", (p[0] + 5, p[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite("bayStatus.jpg", annotated)
    return payload

def format_json_multiline(data):
    lines = ["["]
    for i, item in enumerate(data):
        comma = "," if i < len(data) - 1 else ""
        lines.append(f'  {json.dumps(item, separators=(", ", ": "))}{comma}')
    lines.append("]")
    return "\n".join(lines)

def main():
    json_bays = "bays.json"
    empty_ref = "empty_reference.jpg"
    img_to_process = ""

    demo = False  # set to False to use webcam capture
    interval = 1  # seconds between captures

    while True:
        try:
            if not demo:
                capture_image("current.jpg")
                img_to_process = "current.jpg"
            else:
                img_to_process = "demo.jpg"

            spots_data = analyze_and_prepare_payload(json_bays, empty_ref, img_to_process)

            # Print JSON nicely
            print(format_json_multiline(spots_data))
            print("Saved annotated image as bayStatus.jpg\n")
            url = "http://10.130.1.206:5000/update_spots"  # Replace with your actual endpoint

            try:
                response = requests.post(url, json=spots_data, timeout=5)
                response.raise_for_status()
                print(f"Posted to server: {response.status_code}")
            except requests.RequestException as e:
                print(f"Failed to post to server: {e}")

            time.sleep(interval)

        except Exception as e:
            print("Error during analysis:", e)
            time.sleep(1)

if __name__ == "__main__":
    main()
