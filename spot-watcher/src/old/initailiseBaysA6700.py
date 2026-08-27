#!/usr/bin/env python3
"""
initialiseBaysA6700.py
- Captures a high-res reference photo using the Sony α6700 over USB (gphoto2).
- Resizes frame to standard 1080p height scale while maintaining aspect ratio.
- Detects vertical white line markers to identify parking bay boundaries.
- Clears the './bays/' folder completely prior to saving new crops.
- Saves bay coordinates to 'bays.json' and preview image 'baysDetected.jpg'.
- Crops each detected bay into individual images inside the './bays/' folder.
"""

import json
import os
import shutil
import subprocess
import sys
import cv2
import numpy as np


def capture_reference_photo(
    output_path="emptyReference.jpg", temp_raw="temp_a6700.jpg"
):
    """Triggers the Sony α6700 shutter over USB and prepares a 1080p baseline image."""
    print("Triggering Sony α6700 shutter over USB...")

    cmd = [
        "gphoto2",
        "--capture-image-and-download",
        "--filename",
        temp_raw,
        "--force-overwrite",
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error capturing from Sony α6700 via gphoto2:", file=sys.stderr)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)

    frame = cv2.imread(temp_raw)
    if frame is None:
        print(f"Error: Failed to load captured image '{temp_raw}'.")
        sys.exit(1)

    if os.path.exists(temp_raw):
        os.remove(temp_raw)

    # Resize full image to standard 1080p height scale
    h, w = frame.shape[:2]
    target_h = 1080
    target_w = int(w * (target_h / h))
    resized_frame = cv2.resize(
        frame, (target_w, target_h), interpolation=cv2.INTER_AREA
    )

    print(f"Captured reference image scaled to: {target_w}x{target_h}")
    cv2.imwrite(output_path, resized_frame)
    print(f"Saved empty baseline image to: {output_path}")
    return output_path


def crop_and_save_bays(
    image_path: str = "emptyReference.jpg",
    json_path: str = "bays.json",
    output_dir: str = "./bays",
):
    """Clears output directory, then crops each bay region defined in bays.json into it."""
    if not os.path.exists(json_path) or not os.path.exists(image_path):
        print("Error: Missing JSON or reference image for cropping.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}' for cropping.")
        return

    # Clear directory of any previous run artifacts
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Cleared contents of '{output_dir}/' for a clean setup run.")

    with open(json_path, "r") as f:
        bays = json.load(f)

    print(f"\nCropping {len(bays)} bays into '{output_dir}/'...")

    for bay in bays:
        bay_name = bay["name"]
        pts = bay["points"]

        # Extract bounding box [x1, y1], [x2, y1], [x2, y2], [x1, y2]
        x1, y1 = pts[0]
        x2, y2 = pts[2]

        # Ensure valid pixel slicing coordinates
        crop_y1, crop_y2 = min(y1, y2), max(y1, y2)
        crop_x1, crop_x2 = min(x1, x2), max(x1, x2)

        cropped_bay = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped_bay.size > 0:
            bay_file_path = os.path.join(output_dir, f"{bay_name}.jpg")
            cv2.imwrite(bay_file_path, cropped_bay)
            print(f"Saved crop: {bay_file_path}")
        else:
            print(f"Warning: Empty crop area for {bay_name}")


def detect_parking_bays(
    image_path: str = "emptyReference.jpg",
    output_json: str = "bays.json",
    debug_image_path: str = "baysDetected.jpg",
):
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Could not find baseline image at '{image_path}'"
        )

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image file: {image_path}")

    # Add border padding so edge lines touch black, allowing OpenCV to complete edge contours
    padded_img = cv2.copyMakeBorder(
        img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(cnt)

        if h > 30 and w < 120 and aspect_ratio < 1.2 and area > 150:
            lines.append({"x": x, "y": y, "w": w, "h": h})

    if not lines:
        print(
            "No vertical parking lines detected! Check lighting or thresholding."
        )
        return

    print(f"Found {len(lines)} line markers. Grouping into rows...")

    lines = sorted(lines, key=lambda l: l["y"])
    rows = []
    row_threshold = 120

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

    rows = [row for row in rows if len(row) >= 2]
    rows = sorted(rows, key=lambda r: sum(l["y"] for l in r) / len(r))

    bays = []
    bay_counter = 0
    debug_img = padded_img.copy()

    for row in rows:
        row_sorted = sorted(row, key=lambda l: l["x"])

        for i in range(len(row_sorted) - 1):
            left_line = row_sorted[i]
            right_line = row_sorted[i + 1]

            # Convert back to unpadded original image coordinate space
            x1 = max(0, (left_line["x"] + left_line["w"]) - 20)
            x2 = max(0, right_line["x"] - 20)
            y1 = max(0, min(left_line["y"], right_line["y"]) - 20)
            y2 = max(
                0,
                max(
                    left_line["y"] + left_line["h"],
                    right_line["y"] + right_line["h"],
                )
                - 20,
            )

            bay_width = x2 - x1
            if not (20 <= bay_width <= 700):
                continue

            points = [
                [int(x1), int(y1)],
                [int(x2), int(y1)],
                [int(x2), int(y2)],
                [int(x1), int(y2)],
            ]

            bay_name = f"B{bay_counter}"
            bays.append({"name": bay_name, "points": points})

            pts_np = np.array(
                [[p[0] + 20, p[1] + 20] for p in points], np.int32
            )
            cv2.polylines(debug_img, [pts_np], True, (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                bay_name,
                (left_line["x"] + left_line["w"] + 5, left_line["y"] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            bay_counter += 1

    with open(output_json, "w") as f:
        json.dump(bays, f, indent=2)

    h_orig, w_orig = img.shape[:2]
    debug_img_cropped = debug_img[20 : 20 + h_orig, 20 : 20 + w_orig]
    cv2.imwrite(debug_image_path, debug_img_cropped)

    print(f"Successfully initialized {len(bays)} parking bays.")
    print(f"Saved layout JSON to: {output_json}")
    print(f"Saved setup visualization to: {debug_image_path}")

    # Crop individual bay images directly after detection completes
    crop_and_save_bays(
        image_path=image_path, json_path=output_json, output_dir="./bays"
    )


if __name__ == "__main__":
    print(
        "Running Bay Initialization & Crop Workflow (Sony α6700 USB Mode)...\n"
    )
    ref_image = capture_reference_photo(output_path="emptyReference.jpg")
    detect_parking_bays(image_path=ref_image)