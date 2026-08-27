#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np


def detect_parking_bays(
    image_path: str = "empty_reference.jpg",
    output_json: str = "bays.json",
    debug_image_path: str = "testBaysDetected.jpg",
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find baseline image at '{image_path}'")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image file: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Threshold high to catch the bright white paper strips
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    # 2. Find contours
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(cnt)

        # UPDATED FILTER: For VERTICAL line markers (tall & thin)
        # h > 40 (tall), w < 40 (thin), aspect_ratio < 0.6
        if h > 40 and w < 40 and aspect_ratio < 0.6 and area > 200:
            lines.append({"x": x, "y": y, "w": w, "h": h, "rect": (x, y, w, h)})

    if not lines:
        print("❌ No vertical parking lines detected! Check threshold or filtering.")
        return

    print(f"Detected {len(lines)} line strips. Grouping into rows...")

    # 3. Group lines into horizontal ROWS based on Y position (Top, Middle, Bottom)
    lines = sorted(lines, key=lambda l: l["y"])

    rows = []
    row_threshold = 40  # Max vertical distance to consider lines in the same row

    for line in lines:
        placed = False
        for row in rows:
            avg_y = sum(l["y"] for l in row) / len(row)
            if abs(line["y"] - avg_y) < row_threshold:
                row.append(line)
                placed = True
                break
        if not placed:
            rows.append([line])

    # Keep rows with at least 2 lines
    rows = [row for row in rows if len(row) >= 2]

    # Sort rows from Top to Bottom
    rows = sorted(rows, key=lambda r: sum(l["y"] for l in r) / len(r))

    # 4. Extract spaces BETWEEN adjacent vertical lines in each row as parking bays
    bays = []
    bay_counter = 0

    debug_img = img.copy()

    for row in rows:
        # Sort lines inside the row from Left to Right
        row_sorted = sorted(row, key=lambda l: l["x"])

        for i in range(len(row_sorted) - 1):
            left_line = row_sorted[i]
            right_line = row_sorted[i + 1]

            x1 = left_line["x"] + left_line["w"]
            x2 = right_line["x"]

            y1 = min(left_line["y"], right_line["y"])
            y2 = max(
                left_line["y"] + left_line["h"],
                right_line["y"] + right_line["h"],
            )

            bay_width = x2 - x1

            # Width filter bounds for valid bays
            MAX_BAY_WIDTH = 250
            MIN_BAY_WIDTH = 30

            if not (MIN_BAY_WIDTH <= bay_width <= MAX_BAY_WIDTH):
                continue

            points = [
                [int(x1), int(y1)],  # Top-Left
                [int(x2), int(y1)],  # Top-Right
                [int(x2), int(y2)],  # Bottom-Right
                [int(x1), int(y2)],  # Bottom-Left
            ]

            bay_name = f"B{bay_counter}"
            bays.append({"name": bay_name, "points": points})
            bay_counter += 1

            # Draw green bounding box & text
            pts_np = np.array(points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                debug_img,
                [pts_np],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.putText(
                debug_img,
                bay_name,
                (x1 + 5, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

    # 5. Save output
    with open(output_json, "w") as f:
        json.dump(bays, f, indent=2)

    cv2.imwrite(debug_image_path, debug_img)
    print(
        f"✅ Successfully detected {len(bays)} valid parking bays from '{image_path}'."
    )
    print(f"💾 Saved calibration JSON to: {output_json}")
    print(f"🖼️ Saved debug view to: {debug_image_path}")


if __name__ == "__main__":
    detect_parking_bays(
        image_path="empty_reference.jpg",
        output_json="bays.json",
        debug_image_path="testBaysDetected.jpg",
    )