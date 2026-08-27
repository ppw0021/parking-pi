#!/usr/bin/env python3
"""
initialiseBays.py
- Interactively tunes lens distortion over SSH.
- Saves lens config for other scripts to use.
- Detects vertical white line markers to identify parking bay boundaries.
- Crops each detected bay into individual images inside the './bays/' folder.
"""

import json
import os
import shutil
import sys
import cv2
import numpy as np
import lensCorrection as lens_correction

def apply_post_processing(frame):
    """Applies the cropping and CLAHE lighting balance so the calibration looks exactly like the final image."""
    # Center-crop 1920x1080 to 1440x1080 (cuts off 240px off left & right)
    start_x = (1920 - 1440) // 2
    end_x = start_x + 1440
    cropped_frame = frame[:, start_x:end_x]

    # Balance uneven lighting using CLAHE
    lab = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    balanced_l = clahe.apply(l_channel)
    
    merged_lab = cv2.merge((balanced_l, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def interactive_lens_tuning(camera_index=0):
    print(f"Opening camera (Index {camera_index}) for live calibration...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

    config = lens_correction.load_config()
    k1 = config.get("k1", -0.15)
    k2 = 0.0  
    exposure = config.get("exposure", 150)
    
    # NEW: Software sunglasses (1.0 = normal, 0.5 = half brightness)
    digital_gain = config.get("digital_gain", 1.0)
    
    print("\n--- LIVE CALIBRATION MODE ---")
    print("Controls:")
    print("  '1' / '2' : Lens Bulge (Inward/Outward)")
    print("  '3' / '4' : Hardware Exposure (Darker/Brighter)")
    print("  '5' / '6' : Digital Brightness Multiplier (Darker/Brighter)")
    print("  'y' : Save and continue")
    
    while True:
        cap.set(cv2.CAP_PROP_EXPOSURE, max(0, exposure)) # Prevent negative hardware exposure
        
        for _ in range(5):
            cap.read()
            
        ret, raw_frame = cap.read()
        if not ret or raw_frame is None:
            break

        # NEW: Apply digital darkening BEFORE processing
        darkened_frame = cv2.convertScaleAbs(raw_frame, alpha=digital_gain, beta=0)

        corrected = lens_correction.undistort_frame(darkened_frame, k1, k2)
        processed = apply_post_processing(corrected)
        
        cv2.imwrite("lens_calibration_test.jpg", processed)
        
        print(f"\n> Saved test image.")
        print(f"> Settings: Lens={k1:.2f} | Exp={exposure} | Digital_Gain={digital_gain:.2f}")
        
        user_input = input("Enter command (1-6, or y): ").strip().lower()
        
        if user_input == 'y':
            with open("lens_config.json", "w") as f:
                json.dump({"k1": k1, "k2": k2, "exposure": exposure, "digital_gain": digital_gain}, f, indent=4)
            print("Configuration saved!")
            cv2.imwrite("emptyReference.jpg", processed)
            cap.release()
            return "emptyReference.jpg"
            
        elif user_input == '1': k1 -= 0.05
        elif user_input == '2': k1 += 0.05
        elif user_input == '3': exposure -= 25
        elif user_input == '4': exposure += 25
        elif user_input == '5': digital_gain = max(0.1, digital_gain - 0.1) # Prevent total black screen
        elif user_input == '6': digital_gain += 0.1

def crop_and_save_bays(image_path="emptyReference.jpg", json_path="bays.json", output_dir="./bays"):
    if not os.path.exists(json_path) or not os.path.exists(image_path):
        print("Error: Missing JSON or reference image for cropping.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}' for cropping.")
        return

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

        x1, y1 = pts[0]
        x2, y2 = pts[2]

        crop_y1, crop_y2 = min(y1, y2), max(y1, y2)
        crop_x1, crop_x2 = min(x1, x2), max(x1, x2)

        cropped_bay = img[crop_y1:crop_y2, crop_x1:crop_x2]

        if cropped_bay.size > 0:
            bay_file_path = os.path.join(output_dir, f"{bay_name}.jpg")
            cv2.imwrite(bay_file_path, cropped_bay)
            print(f"Saved crop: {bay_file_path}")
        else:
            print(f"Warning: Empty crop area for {bay_name}")


def detect_parking_bays(image_path="emptyReference.jpg", output_json="bays.json", debug_image_path="baysDetected.jpg"):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Could not find baseline image at '{image_path}'")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image file: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h if h > 0 else 0
        area = cv2.contourArea(cnt)

        if h > 40 and w < 100 and aspect_ratio < 0.8 and area > 200:
            lines.append({"x": x, "y": y, "w": w, "h": h})

    if not lines:
        print("No vertical parking lines detected!")
        return

    print(f"Found {len(lines)} line markers. Grouping into rows...")
    lines = sorted(lines, key=lambda l: l["y"])
    rows = []
    row_threshold = 80

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

    candidate_bays = []
    for row in rows:
        row_sorted = sorted(row, key=lambda l: l["x"])
        for i in range(len(row_sorted) - 1):
            left_line = row_sorted[i]
            right_line = row_sorted[i + 1]

            if abs(left_line["y"] - right_line["y"]) > 50:
                continue
            if abs(left_line["h"] - right_line["h"]) > 50:
                continue

            x1 = left_line["x"] + left_line["w"]
            x2 = right_line["x"]
            y1 = min(left_line["y"], right_line["y"])
            y2 = max(left_line["y"] + left_line["h"], right_line["y"] + right_line["h"])

            bay_width = x2 - x1
            bay_height = y2 - y1

            if bay_width >= bay_height:
                continue

            candidate_bays.append({
                "points": [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]],
                "width": bay_width,
                "height": bay_height,
                "x1": x1,
                "y1": y1
            })

    bays = []
    debug_img = img.copy()  # <--- Moved this outside the if statement!
    
    if candidate_bays:
        # Calculate the median dimensions of all valid candidate bays
        median_w = np.median([b["width"] for b in candidate_bays])
        median_h = np.median([b["height"] for b in candidate_bays])

        bay_counter = 0

        for b in candidate_bays:
            w_err = abs(b["width"] - median_w) / median_w
            h_err = abs(b["height"] - median_h) / median_h
            
            if w_err < 0.30 and h_err < 0.30:
                bay_name = f"B{bay_counter}"
                
                # --- NEW: Scale the bay inward to avoid the white lines ---
                scale_factor = 0.80
                pts_float = np.array(b["points"], np.float32)
                centroid = np.mean(pts_float, axis=0)
                scaled_pts = centroid + (pts_float - centroid) * scale_factor
                final_points = scaled_pts.astype(np.int32).tolist()
                # ----------------------------------------------------------

                bays.append({"name": bay_name, "points": final_points})

                # Draw the scaled box on the debug image
                pts_np = np.array(final_points, np.int32)
                cv2.polylines(debug_img, [pts_np], True, (0, 255, 0), 2)
                
                # Put text near the top-left of the new scaled box
                text_x, text_y = final_points[0]
                cv2.putText(
                    debug_img,
                    bay_name,
                    (text_x + 5, text_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                bay_counter += 1

    with open(output_json, "w") as f:
        json.dump(bays, f, indent=2)

    cv2.imwrite(debug_image_path, debug_img)
    print(f"\nSuccessfully initialized {len(bays)} perfectly sized parking bays.")
    print(f"Saved layout JSON to: {output_json}")
    print(f"Saved setup visualization to: {debug_image_path}")

    crop_and_save_bays(image_path=image_path, json_path=output_json, output_dir="./bays")


if __name__ == "__main__":
    print("Running Bay Initialization Workflow (1080p Lens Corrected)...\n")
    # 1. Run the interactive calibration over SSH
    ref_image = interactive_lens_tuning()
    
    # 2. Proceed with bay detection using the calibrated baseline
    detect_parking_bays(image_path=ref_image)