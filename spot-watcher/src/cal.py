#!/usr/bin/env python3
"""
cal.py - calibrate the 16 parking-bay boxes for a headless (SSH) Pi.

The lot is always 6 bays in the top row, 4 in the middle, 6 in the bottom
(spot ids 0-5, 6-9, 10-15 - the order the web server expects).

A box is four pixel numbers: x, y, w, h. There is no GUI: you look at
`cal_preview.jpg`, adjust numbers, and look again.

Workflow
--------
1.  uv run cal.py
    First run tries to auto-detect the bays from the bright lane markers and
    fit the 6/4/6 layout; if that fails it falls back to an even grid. Either
    way it writes spots.json and cal_preview.jpg (plus cal_auto_debug.jpg
    showing the detected markers).
2.  Open cal_preview.jpg (VS Code Remote, scp, shared folder) and compare
    the numbered boxes against the real bays.
3.  At the `cal>` prompt, adjust until every box sits on its bay:
      auto                     re-run marker auto-detection
      <id> <x> <y> <w> <h>     move one box            e.g.  5 120 340 90 160
      row<0|1|2> <x> <y> <w> <h>   set a row's outer rectangle, split evenly
      mv <id> <dx> <dy>        nudge one box
      grid                     reset to the default 6/4/6 grid
      show                     print the current boxes
      shot                     just re-grab the frame and redraw the preview
      save                     capture the empty-lot reference crops (refs/)
      q                        quit
    Every change rewrites spots.json and cal_preview.jpg.
4.  With the lot empty and the boxes aligned, run `save`. It fills
    refs/00.png .. 15.png and writes cal_preview_refs.jpg to check.
"""
import json
import os
import statistics
import sys

import cv2
import numpy as np

# ---------------- Configuration ----------------
CAMERA_INDEX = 0
RESOLUTION = (1920, 1080)          # (width, height) requested from the camera
SPOTS_PATH = "spots.json"
REFS_DIR = "refs"
PREVIEW_PATH = "cal_preview.jpg"
REFS_PREVIEW_PATH = "cal_preview_refs.jpg"
AUTO_DEBUG_PATH = "cal_auto_debug.jpg"
REF_SIZE = (128, 128)             # every reference crop is stored at this size

# (first spot id, number of spots) for the top, middle and bottom rows
ROWS = [(0, 6), (6, 4), (10, 6)]

# Auto-detection: bays are read from bright, tall-thin vertical lane markers.
MARKER_THRESH = 200               # grayscale level a marker pixel must exceed
MARKER_MIN_H_FRAC = 0.04          # min marker height, fraction of frame height
MARKER_MAX_W_FRAC = 0.05          # max marker width, fraction of frame width
MARKER_MIN_ASPECT = 2.0          # min height / width of a marker blob
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
    """Median of a few fresh frames - kills sensor noise in the references."""
    frames = []
    for _ in range(7):
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        sys.exit("Camera returned no frames")
    return np.median(np.stack(frames[-5:]), axis=0).astype(np.uint8)


def default_grid(width, height):
    """A first guess: 6/4/6 boxes spread across the frame."""
    # (y0, y1) band and (x0, x1) span for each row, as fractions of the frame
    bands = [(0.12, 0.36, 0.05, 0.95),
             (0.42, 0.58, 0.20, 0.80),
             (0.64, 0.88, 0.05, 0.95)]
    spots = {}
    for (start_id, count), (y0, y1, x0, x1) in zip(ROWS, bands):
        row_x, row_y = x0 * width, y0 * height
        row_w, row_h = (x1 - x0) * width, (y1 - y0) * height
        spots.update(split_row(start_id, count, row_x, row_y, row_w, row_h))
    return spots


def split_row(start_id, count, x, y, w, h, gap_frac=0.15):
    """Divide an outer rectangle into `count` evenly spaced boxes."""
    cell = w / count
    gap = cell * gap_frac
    boxes = {}
    for i in range(count):
        boxes[str(start_id + i)] = [
            int(round(x + i * cell + gap / 2)),
            int(round(y)),
            int(round(cell - gap)),
            int(round(h)),
        ]
    return boxes


def find_markers(frame):
    """Bright, tall-thin vertical blobs -> list of (x, y, w, h) bounding rects."""
    h_img, w_img = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, MARKER_THRESH, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(5, int(h_img * 0.03))))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h >= MARKER_MIN_H_FRAC * h_img
                and w <= MARKER_MAX_W_FRAC * w_img
                and h / max(w, 1) >= MARKER_MIN_ASPECT):
            markers.append((x, y, w, h))
    return markers


def group_rows(markers):
    """Cluster markers into horizontal rows by vertical centre."""
    if not markers:
        return []
    med_h = statistics.median(m[3] for m in markers)
    tol = max(med_h * 0.7, 25)
    rows = []
    for m in sorted(markers, key=lambda m: m[1]):
        cy = m[1] + m[3] / 2
        for row in rows:
            avg = sum(r[1] + r[3] / 2 for r in row) / len(row)
            if abs(cy - avg) < tol:
                row.append(m)
                break
        else:
            rows.append([m])
    return rows


def auto_detect(frame):
    """Detect lane markers and fit the 6/4/6 layout.

    Returns (spots_dict_or_None, markers). When markers line up as exactly
    `count + 1` per row the bay edges are taken from them; otherwise the row
    is split evenly between its outermost markers.
    """
    h_img, w_img = frame.shape[:2]
    markers = find_markers(frame)
    rows = [r for r in group_rows(markers) if len(r) >= 2]
    if len(rows) < 3:
        return None, markers
    rows = sorted(rows, key=len, reverse=True)[:3]
    rows.sort(key=lambda r: sum(m[1] for m in r) / len(r))

    spots = {}
    for (start_id, count), row in zip(ROWS, rows):
        row = sorted(row, key=lambda m: m[0])
        y0 = min(m[1] for m in row)
        y1 = max(m[1] + m[3] for m in row)
        if len(row) == count + 1:
            edges = [m[0] + m[2] / 2 for m in row]
            for i in range(count):
                lx, rx = edges[i], edges[i + 1]
                gap = (rx - lx) * 0.12
                spots[str(start_id + i)] = clamp_box(
                    [lx + gap, y0, (rx - lx) - 2 * gap, y1 - y0], w_img, h_img)
        else:
            x0 = min(m[0] for m in row)
            x1 = max(m[0] + m[2] for m in row)
            spots.update({
                k: clamp_box(v, w_img, h_img) for k, v in
                split_row(start_id, count, x0, y0, x1 - x0, y1 - y0).items()})

    if len(spots) != 16:
        return None, markers
    return spots, markers


def draw_auto_debug(frame, markers, spots):
    vis = frame.copy()
    for x, y, w, h in markers:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
    for i in range(16):
        box = spots.get(str(i))
        if box:
            x, y, w, h = box
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite(AUTO_DEBUG_PATH, vis)


def try_auto(frame):
    """Run auto_detect, write the debug image, print a summary. Returns spots or None."""
    spots, markers = auto_detect(frame)
    draw_auto_debug(frame, markers, spots or {})
    if spots:
        print(f"auto-detect: {len(markers)} markers -> 16 boxes "
              f"(check {AUTO_DEBUG_PATH})")
    else:
        print(f"auto-detect failed: {len(markers)} markers, need 3 rows of >=2 "
              f"(check {AUTO_DEBUG_PATH}); tune MARKER_* in cal.py")
    return spots


def row_for_id(spot_id):
    for start_id, count in ROWS:
        if start_id <= spot_id < start_id + count:
            return start_id, count
    return None


def load_spots():
    with open(SPOTS_PATH) as f:
        data = json.load(f)
    return data.get("spots", {})


def save_spots(spots):
    ordered = {str(i): spots[str(i)] for i in range(16) if str(i) in spots}
    with open(SPOTS_PATH, "w") as f:
        json.dump({"resolution": list(RESOLUTION), "spots": ordered}, f, indent=2)


def clamp_box(box, width, height):
    x, y, w, h = box
    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))
    w = max(1, min(int(w), width - x))
    h = max(1, min(int(h), height - y))
    return [x, y, w, h]


def draw_preview(frame, spots, path):
    vis = frame.copy()
    for i in range(16):
        box = spots.get(str(i))
        if not box:
            continue
        x, y, w, h = box
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, str(i), (x + 4, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imwrite(path, vis)
    print(f"wrote {path}")


def capture_refs(frame, spots):
    os.makedirs(REFS_DIR, exist_ok=True)
    missing = [str(i) for i in range(16) if str(i) not in spots]
    if missing:
        print(f"refusing to save: boxes {', '.join(missing)} are not set")
        return
    tiles = []
    for i in range(16):
        x, y, w, h = spots[str(i)]
        crop = frame[y:y + h, x:x + w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, REF_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(REFS_DIR, f"{i:02d}.png"), gray)
        tile = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.putText(tile, str(i), (4, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(tile)
    grid = np.vstack([np.hstack(tiles[r:r + 8]) for r in (0, 8)])
    cv2.imwrite(REFS_PREVIEW_PATH, grid)
    print(f"saved 16 references to {REFS_DIR}/ and wrote {REFS_PREVIEW_PATH}")


def handle(cmd, spots, frame):
    """Mutate `spots` in place. Return True if the frame should be re-grabbed."""
    parts = cmd.split()
    if not parts:
        return False
    head, args = parts[0], parts[1:]
    h_img, w_img = frame.shape[:2]

    if head == "auto":
        found = try_auto(frame)
        if found:
            spots.clear()
            spots.update(found)
    elif head == "grid":
        spots.clear()
        spots.update(default_grid(w_img, h_img))
    elif head == "show":
        for i in range(16):
            print(f"  {i:2d}: {spots.get(str(i))}")
    elif head == "shot":
        return True
    elif head == "mv" and len(args) == 3:
        i, dx, dy = args
        box = spots.get(i)
        if box:
            box[0] += int(dx)
            box[1] += int(dy)
            spots[i] = clamp_box(box, w_img, h_img)
    elif head.startswith("row") and len(args) == 4:
        try:
            row_idx = int(head[3:])
            start_id, count = ROWS[row_idx]
        except (ValueError, IndexError):
            print("row index must be 0, 1 or 2")
            return False
        x, y, w, h = (float(a) for a in args)
        spots.update({k: clamp_box(v, w_img, h_img)
                      for k, v in split_row(start_id, count, x, y, w, h).items()})
    elif head.isdigit() and len(args) == 4:
        i = int(head)
        if 0 <= i <= 15:
            spots[str(i)] = clamp_box([float(a) for a in args], w_img, h_img)
        else:
            print("spot id must be 0-15")
    else:
        print("commands: auto | <id> x y w h | row<0-2> x y w h | mv <id> dx dy "
              "| grid | show | shot | save | q")
    return False


def main():
    cap = open_camera()
    frame = grab_frame(cap)
    h_img, w_img = frame.shape[:2]
    print(f"camera frame is {w_img}x{h_img}")

    if os.path.exists(SPOTS_PATH):
        spots = load_spots()
        print(f"loaded {len(spots)} boxes from {SPOTS_PATH}")
    else:
        spots = try_auto(frame)
        if not spots:
            spots = default_grid(w_img, h_img)
            print(f"seeded {SPOTS_PATH} with a default 6/4/6 grid")
        save_spots(spots)

    draw_preview(frame, spots, PREVIEW_PATH)
    print(__doc__.split("Workflow")[0].strip())

    while True:
        try:
            cmd = input("cal> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "save":
            capture_refs(frame, spots)
            continue
        regrab = handle(cmd, spots, frame)
        if regrab:
            frame = grab_frame(cap)
        save_spots(spots)
        draw_preview(frame, spots, PREVIEW_PATH)

    cap.release()


if __name__ == "__main__":
    main()
