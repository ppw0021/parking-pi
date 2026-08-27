#!/usr/bin/env python3
"""
Test script mirroring live webcam startup workflow:
1. AT STARTUP (ONCE): Reads 'testEmpty.jpg', detects bay boundaries, 
   saves 'bays.json', AND saves 'testBaysDetected.jpg' preview.
2. IN LOOP: Reads 'testCurrent.jpg', checks occupancy against 'testEmpty.jpg', 
   prints JSON, saves 'testBayStatus.jpg', and POSTs to server.
"""

import json
import os
import re
import time
import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim

# -------- thresholds --------
SSIM_MIN = 0.85
SAT_DELTA_MIN = 50.0
LUMA_DELTA_MIN = 50.0
CENTER_SIGMA_FRAC = 0.25
RESIZE_TO = (120, 120)
# ----------------------------


# ==================== STEP 1: INITIALIZATION (ONCE AT STARTUP) ==================== #
def initialize_parking_bays(empty_img_path="testEmpty.jpg", output_json="bays.json", preview_img_path="testBaysDetected.jpg"):
    """
    Runs ONCE at startup:
    Reads the empty image, detects parking lines, saves coordinates to bays.json,
    and exports 'testBaysDetected.jpg' as a visual verification.
    """
    print(f"⚙️ Initializing: Detecting parking bays on empty frame '{empty_img_path}'...")

    if not os.path.exists(empty_img_path):
        raise FileNotFoundError(
            f"❌ Initialization failed: '{empty_img_path}' not found."
        )

    img = cv2.imread(empty_img_path)
    if img is None:
        raise ValueError(f"❌ Failed to load image '{empty_img_path}'.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(cnt)

        # Filter horizontal parking lines
        if w > 40 and h < 35 and aspect_ratio > 2.0 and area > 300:
            lines.append({"x": x, "y": y, "w": w, "h": h})

    if not lines:
        print("⚠️ Warning: No parking lines detected on the empty reference frame!")
        return False

    # Group lines into vertical columns based on X coordinate
    lines = sorted(lines, key=lambda l: l["x"])
    columns = []
    col_threshold = 50

    for line in lines:
        placed = False
        for col in columns:
            avg_x = sum(l["x"] for l in col) / len(col)
            if abs(line["x"] - avg_x) < col_threshold:
                col.append(line)
                placed = True
                break
        if not placed:
            columns.append([line])

    columns = [col for col in columns if len(col) >= 2]
    columns = sorted(columns, key=lambda col: sum(l["x"] for l in col) / len(col))

    bays = []
    bay_counter = 0
    annotated_empty = img.copy()

    for col in columns:
        col_sorted = sorted(col, key=lambda l: l["y"])

        for i in range(len(col_sorted) - 1):
            top_line = col_sorted[i]
            bottom_line = col_sorted[i + 1]

            x1 = min(top_line["x"], bottom_line["x"])
            x2 = max(
                top_line["x"] + top_line["w"], bottom_line["x"] + bottom_line["w"]
            )
            y1 = top_line["y"] + top_line["h"]
            y2 = bottom_line["y"]

            bay_height = y2 - y1
            if not (30 <= bay_height <= 200):
                continue

            points = [
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)],
            ]

            bays.append({"name": f"B{bay_counter}", "points": points})

            # Draw green box & ID on empty setup preview image
            pts = np.array(points, np.int32)
            cv2.polylines(annotated_empty, [pts], True, (0, 255, 0), 2)
            cv2.putText(
                annotated_empty,
                f"{bay_counter}",
                (points[0][0] + 5, points[0][1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            bay_counter += 1

    # Save output JSON
    with open(output_json, "w") as f:
        json.dump(bays, f, indent=2)

    # Save visual confirmation frame as testBaysDetected.jpg
    cv2.imwrite(preview_img_path, annotated_empty)

    print(f"✅ Startup Complete!")
    print(f"   • Detected {len(bays)} bays on '{empty_img_path}'")
    print(f"   • Saved bounding coordinates to '{output_json}'")
    print(f"   • Saved setup preview to '{preview_img_path}'\n")
    return True


# ==================== STEP 2: OCCUPANCY ANALYSIS ==================== #
def load_bays(path):
    with open(path, "r") as f:
        return json.load(f)


def crop_polygon(img, points):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    x, y, w, h = cv2.boundingRect(pts)
    roi = cv2.bitwise_and(img, img, mask=mask)[y : y + h, x : x + w]
    return roi


def center_weight(h, w, sigma_frac=CENTER_SIGMA_FRAC):
    y, x = np.ogrid[:h, :w]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    sigma = min(w, h) * sigma_frac
    wgt = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
    return wgt / (wgt.mean() + 1e-9)


def patch_metrics(ref_patch, cur_patch):
    ref = cv2.resize(ref_patch, RESIZE_TO)
    cur = cv2.resize(cur_patch, RESIZE_TO)
    ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    cur_hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    ref_v, ref_s = (
        ref_hsv[..., 2].astype(np.float32),
        ref_hsv[..., 1].astype(np.float32),
    )
    cur_v, cur_s = (
        cur_hsv[..., 2].astype(np.float32),
        cur_hsv[..., 1].astype(np.float32),
    )

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
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else fallback


def analyze_and_prepare_payload(bays_path, ref_path, img_path, status_img_path="testBayStatus.jpg"):
    bays = load_bays(bays_path)
    ref = cv2.imread(ref_path)
    cur = cv2.imread(img_path)
    if ref is None or cur is None:
        raise SystemExit(
            f"Could not read reference image ({ref_path}) or current image ({img_path})"
        )

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

        s_w, d_sat, d_luma = temp[name]
        adj_luma = abs(d_luma)

        taken = (
            (s_w < SSIM_MIN)
            or (d_sat > SAT_DELTA_MIN)
            or (adj_luma > LUMA_DELTA_MIN)
        )

        bay_id = parse_id_from_name(name, fallback_id)
        fallback_id += 1
        payload.append({"id": bay_id, "taken": bool(taken)})

        color = (0, 0, 255) if taken else (0, 255, 0)
        pts = np.array(b["points"], np.int32)
        cv2.polylines(annotated, [pts], True, color, 2)
        p = pts[0]
        cv2.putText(
            annotated,
            f"{bay_id}",
            (p[0] + 5, p[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.imwrite(status_img_path, annotated)
    return payload


def format_json_multiline(data):
    lines = ["["]
    for i, item in enumerate(data):
        comma = "," if i < len(data) - 1 else ""
        lines.append(f'  {json.dumps(item, separators=(", ", ": "))}{comma}')
    lines.append("]")
    return "\n".join(lines)


# ==================== MAIN EXECUTION ==================== #
def main():
    json_bays = "bays.json"
    empty_ref = "testEmpty.jpg"
    current_img = "testCurrent.jpg"
    preview_img = "testBaysDetected.jpg"
    status_img = "testBayStatus.jpg"
    interval = 2  # Seconds between loop cycles

    print("🚀 Running testMain.py...\n")

    # --- 1. RUN ONCE AT STARTUP ---
    # Detects bays from testEmpty.jpg, generates bays.json & testBaysDetected.jpg
    initialize_parking_bays(empty_img_path=empty_ref, output_json=json_bays, preview_img_path=preview_img)

    # --- 2. RUN OCCUPANCY LOOP ---
    print("🔁 Starting occupancy loop (evaluating testCurrent.jpg against testEmpty.jpg)...")

    while True:
        try:
            if not os.path.exists(current_img):
                print(f"⚠️ Waiting for '{current_img}' to exist...")
                time.sleep(2)
                continue

            # Check occupancy and save output as testBayStatus.jpg
            spots_data = analyze_and_prepare_payload(
                json_bays, empty_ref, current_img, status_img_path=status_img
            )

            # Print JSON status
            print(format_json_multiline(spots_data))
            print(f"💾 Updated {status_img}\n")

            # POST payload to Flask
            url = "http://10.130.1.206:5000/update_spots"
            try:
                response = requests.post(url, json=spots_data, timeout=5)
                response.raise_for_status()
                print(f"📡 Posted to server: {response.status_code}\n")
            except requests.RequestException as e:
                print(f"⚠️ Server connection error: {e}\n")

            time.sleep(interval)

        except Exception as e:
            print("❌ Loop Error:", e)
            time.sleep(2)


if __name__ == "__main__":
    main()