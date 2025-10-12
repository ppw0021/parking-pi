'''
GateWatcher:
The code will monitor the gates of a parking lot.
If a license plate is detected on the right part of the creen,
It means a car is entering, so the code will issue a call to the server:
    URL/enter/<plate>
The server will reply with one of a following status_codes:
    110 - if the car was added correctly to the database.
    111 - if failed, e.g. car is already in the carpark.

If the plate is detected on the left part of the screen,
It means the car is exiting the parking lot, so the call will be:
    URL/exit/<plate>
The server will reply with one of the following status_codes:
    120 - if the fees for the parking were paid for this car
    121 - if the fees for the parking were NOT paid for this car
    122 - if any error occured
'''

import cv2
import numpy as np
import pytesseract
import requests
import time
from datetime import datetime

# ---- Configuration --------------------------------------------------------

CAMERA_INDEX = 0                 # Ususaly '0' for the first connected camera
WEB_PI_IP = "http://127.0.0.1"  # Holds the IP of the web server pi
URL = f"{WEB_PI_IP}:5000"        # Holds the full address of the server

MIN_AREA = 2000                  # 
ASPECT_MIN = 2.0                 # 
ASPECT_MAX = 6.0                 # 
MAX_CANDIDATES = 6               # 
TESS_CFG = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PRINT_ALL_OCR = True

# ---- Functions -------------------------------------------------------------

def preprocess_roi(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 7, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(filt)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return th


def find_plate_candidates(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 80, 200)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:MAX_CANDIDATES]

def ocr_plate(roi_bin):
    txt = pytesseract.image_to_string(roi_bin, config=TESS_CFG)
    return txt.strip()

def normalize_plate(txt):
    s = txt.upper().strip().replace(" ", "")
    s = s.replace("O", "0").replace("I", "1").replace("Z", "2")
    s = s.replace("S", "5").replace("B", "8")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(ch for ch in s if ch in allowed)

def send_plate_event(plate, x, frame_width):
    try:
        if x > frame_width * 0.6:
            uri = f"{URL}/enter/{plate}"
        elif x < frame_width * 0.4:
            uri = f"{URL}/exit/{plate}"
        else:
            return  # ignore center

        response = requests.get(uri, timeout=5)
        print(f"Sent {uri} → Status: {response.status_code}")
        if response.status_code == 110:
            print("Car entered successfully.")
        elif response.status_code == 111:
            print("Car already in carpark.")
        elif response.status_code == 120:
            print("Exit: fees paid.")
        elif response.status_code == 121:
            print("Exit: fees NOT paid.")
        elif response.status_code == 122:
            print("Exit: error occurred.")
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

# ---- Main Loop ---------------------------------------------------------

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        return
    print("Press 'q' to quit, 's' to save snapshot.")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue

        boxes = find_plate_candidates(frame)
        labels = []

        for idx, (x,y,w,h) in enumerate(boxes,1):
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, None, fx=2.0, fy=2.0,
                             interpolation=cv2.INTER_CUBIC)
            th = preprocess_roi(roi)
            raw = ocr_plate(th)
            plate = normalize_plate(raw)

            if PRINT_ALL_OCR:
                print(f"[cand {idx}] raw={raw!r} norm={plate!r}")

            if 5 <= len(plate) <= 8:
                send_plate_event(plate, x, frame.shape[1])
                labels.append(plate)
            else:
                labels.append("")

        vis = frame.copy()
        for (x,y,w,h), lab in zip(boxes, labels):
            cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0),2)
            if lab:
                cv2.putText(vis, lab, (x,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,
                            (0,255,0), 2, cv2.LINE_AA)
                
        cv2.imshow("Gate Watcher", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting")
            break
        if key == ord('s'):
            name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
            cv2.imweitw(name, vis)
            print(f"Frame saved into {name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()