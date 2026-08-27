#!/usr/bin/env python3
"""
main.py - watch the 16 parking bays and report occupancy to the web server.

Needs calibration data from cal.py: spots.json + refs/00.png .. 15.png.
Each loop it crops every bay from a fresh frame, compares it against that
bay's empty-lot reference, and POSTs the result to {SERVER_URL}/update_spots
as [{"id": 0, "taken": false}, ...].

    uv run main.py              run against the web server
    uv run main.py --dry-run    print per-bay scores, write dryrun_preview.jpg,
                                do not contact the server (use this to pick
                                DIFF_THRESHOLD)
"""
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

# ---------------- Configuration ----------------
SERVER_URL = "http://10.0.0.2:5000"   # web-server address - set this per site
CAMERA_INDEX = 0
RESOLUTION = (1920, 1080)
INTERVAL_SEC = 1.0

SPOTS_PATH = "spots.json"
REFS_DIR = "refs"
REF_SIZE = (128, 128)
DRYRUN_PREVIEW_PATH = "dryrun_preview.jpg"

# A bay counts as "taken" when more than DIFF_THRESHOLD of its pixels changed
# by more than PIXEL_DELTA (0-255) versus the empty reference. Global lighting
# shifts up to BRIGHT_TOLERANCE levels are compensated first so slow daylight
# drift does not read as a car.
PIXEL_DELTA = 25
DIFF_THRESHOLD = 0.10
BRIGHT_TOLERANCE = 18
# ----------------------------------------------


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION[1])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def grab_frame(cap):
    frame = None
    for _ in range(5):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    if frame is None:
        raise RuntimeError("camera returned no frames")
    return frame


def load_calibration():
    if not os.path.exists(SPOTS_PATH):
        sys.exit(f"{SPOTS_PATH} missing - run cal.py first")
    with open(SPOTS_PATH) as f:
        data = json.load(f)
    spots = data.get("spots", {})
    if len(spots) != 16:
        sys.exit(f"{SPOTS_PATH} has {len(spots)} boxes, expected 16 - re-run cal.py")

    refs = {}
    for i in range(16):
        path = os.path.join(REFS_DIR, f"{i:02d}.png")
        ref = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if ref is None:
            sys.exit(f"missing reference {path} - run `save` in cal.py")
        refs[i] = cv2.resize(ref, REF_SIZE, interpolation=cv2.INTER_AREA)

    cal_res = data.get("resolution")
    if cal_res and tuple(cal_res) != tuple(RESOLUTION):
        print(f"warning: calibrated at {cal_res}, running at {list(RESOLUTION)}")
    return spots, refs


def prep(crop):
    """Bay crop -> fixed-size, blurred, grayscale patch ready for diffing."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, REF_SIZE, interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def score_bay(patch, ref):
    """Fraction of pixels that changed vs the reference, after compensating a
    small global lighting shift (a car changes far more than BRIGHT_TOLERANCE
    over much of the box; daylight drift changes a little, evenly)."""
    shift = float(ref.mean()) - float(patch.mean())
    shift = max(-BRIGHT_TOLERANCE, min(BRIGHT_TOLERANCE, shift))
    matched = np.clip(patch.astype(np.int16) + shift, 0, 255).astype(np.uint8)
    diff = cv2.absdiff(matched, ref)
    return float(np.mean(diff > PIXEL_DELTA))


def evaluate(frame, spots, refs):
    scores, statuses = {}, {}
    for i in range(16):
        x, y, w, h = spots[str(i)]
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            scores[i], statuses[i] = 0.0, False
            continue
        s = score_bay(prep(crop), refs[i])
        scores[i] = s
        statuses[i] = s > DIFF_THRESHOLD
    return scores, statuses


def post(statuses):
    payload = [{"id": i, "taken": bool(statuses[i])} for i in range(16)]
    try:
        r = requests.post(f"{SERVER_URL}/update_spots", json=payload, timeout=2)
        if r.status_code != 200:
            print(f"server replied {r.status_code}: {r.text.strip()[:120]}")
    except requests.RequestException as e:
        print(f"post failed: {e}")


def bitmap(statuses):
    return "".join("X" if statuses[i] else "." for i in range(16))


def dry_run(cap, spots, refs):
    print("dry run - Ctrl+C to stop\n")
    while True:
        scores, statuses = evaluate(grab_frame(cap), spots, refs)
        print(time.strftime("%H:%M:%S"), bitmap(statuses))
        for r in (0, 8):
            print("  " + "  ".join(
                f"{i:2d}:{scores[i]:.3f}{'*' if statuses[i] else ' '}"
                for i in range(r, r + 8)))
        annotate(grab_frame(cap), spots, scores, statuses)
        time.sleep(INTERVAL_SEC)


def annotate(frame, spots, scores, statuses):
    vis = frame.copy()
    for i in range(16):
        x, y, w, h = spots[str(i)]
        color = (0, 0, 255) if statuses[i] else (0, 255, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, f"{i}:{scores[i]:.2f}", (x + 4, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(DRYRUN_PREVIEW_PATH, vis)


def main():
    dry = "--dry-run" in sys.argv[1:]
    spots, refs = load_calibration()
    cap = open_camera()
    try:
        if dry:
            dry_run(cap, spots, refs)
            return
        print(f"reporting to {SERVER_URL} every {INTERVAL_SEC}s - Ctrl+C to stop")
        while True:
            t0 = time.time()
            try:
                _, statuses = evaluate(grab_frame(cap), spots, refs)
            except RuntimeError as e:
                print(f"frame error: {e}")
                time.sleep(0.5)
                continue
            post(statuses)
            print(time.strftime("%H:%M:%S"), bitmap(statuses),
                  f"({sum(statuses.values())}/16 taken)")
            time.sleep(max(0.0, INTERVAL_SEC - (time.time() - t0)))
    except KeyboardInterrupt:
        print()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
