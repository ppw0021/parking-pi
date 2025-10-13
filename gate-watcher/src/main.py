import requests
import servo # servo.py
from time import sleep
import cv2
import numpy as np
import pytesseract
import time
from datetime import datetime

'''
GateWatcher:
The code monitors the gates of a parking lot.
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

Antispam feature added:
    Read OCR every 500 milliseconds
    Block similar/partial re-sends for 10 seconds after last send
    If still the same number after timeout - print "{plate}, please move on"
    If after timeout the number is similar - send the new plate
'''

# Replace with your target IP (and include http:// or https://)
url = "http://127.0.0.1:5000"

try:
    plate = "54"
    uri = url + "/exit/" + plate
    response = requests.get(uri, timeout=5)
    print(f"Status code: {response.status_code}")

    if (response.status_code == 201):
        print("Customer has paid, open the gate")
        # customer has paid open the gates, blink LED whatever
    elif (response.status_code == 202):
        print("Customer has not paid, do not open gate")
        # Customer has not paid, blink red LED
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")


# Open Gates
# servo.set_gate(0, False)  # Open gate 0 (Entry gate)
# servo.set_gate(1, False)  # Open gate 1 (Exit gate)
# sleep(1)

#Close Gates
# servo.set_gate(0, True)   # Close gate 0 (Entry gate)
# servo.set_gate(1, True)   # Close gate 1 (Exit gate)

# ---- Configuration ---------------------------------------------------------

CAMERA_INDEX = 0                 # Ususaly '0' for the first connected camera
WEB_PI_IP = "http://127.0.0.1"  # Holds the IP of the web server pi
URL = f"{WEB_PI_IP}:5000"        # Holds the full address of the server

MIN_AREA = 2000              # 
ASPECT_MIN = 2.0             # 
ASPECT_MAX = 6.0             # 
MAX_CANDIDATES = 6           # 
TESS_CFG = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PRINT_ALL_OCR = True

# OCR constants:
EXIT_ZONE_X_LIMIT  = 0.4     # 
ENTER_ZONE_X_LIMIT = 0.6     # 

# Anti-Spam controls:
READ_PERIOD = 0.5            # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0       # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1     # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1   # Allow one missing char in a prefix match

READ_PERIOD = 0.5           # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0      # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1    # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1  # Allow one missing char in a prefix match

# Global Anti-spam state variables:
last_sent_plate = ""        # last plate sent to server
block_active = False        # are we in 10 s block window?
block_deadline = 0.0        # monotonic deadline when block expires

last_seen_plate = ""        # most recently observed candidate
last_seen_x = 0             # x coordinate of last seen candidate
last_seen_time = 0.0        # monotonic time when we saw last candidate
last_seen_equal = False     # whether last seen equals last_sent_plate

next_read_ts = 0.0          # throttle OCR (monotonic time)

# ---- Functions -------------------------------------------------------------
# -------- Similarity --------------------------------------------------------

def levenshtein(a, b):
    """
    Function: levenshtein
    Purpose: Compute edit distance between strings a and b.
    Methods: Dynamic programming over a matrix of size len(a) x len(b).
    Creates: dp 2D list for distances, local indices i, j.
    """
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost # substitution
            )
    return dp[-1][-1]


def is_partial_or_similar(a, b):
    """
    Function: is_partial_or_similar
    Purpose: Decide if two plates are equal, very close, or partial.
    Methods: Exact check, Levenshtein distance, prefix containment.
    Creates: d distance int, min_len int, allow_drop int.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    d = levenshtein(a, b)
    if d <= SIMILAR_DISTANCE_MAX:
        return True
    min_len = min(len(a), len(b))
    allow_drop = PARTIAL_MIN_MATCH_DROP
    if a[:min_len - allow_drop] == b[:min_len - allow_drop]:
        return True
    if (a in b) or (b in a):
        short = min(len(a), len(b))
        overlap = min(len(a), len(b)) - allow_drop
        return overlap >= (short - allow_drop)
    return False

# -------- Image preprocessing and OCR ---------------------------------------
def preprocess_roi(roi_bgr):
    """
    Function: preprocess_roi
    Purpose: Prepare a candidate plate region for OCR.
    Methods: Convert to gray, bilateral filter, CLAHE, Otsu threshold.
    Creates: gray, filt, clahe, eq, th local variables.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 7, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(filt)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    return th


def find_plate_candidates(frame_bgr):
    """
    Function: find_plate_candidates
    Purpose: Detect rectangular regions that may contain a plate.
    Methods: Gray, blur, Canny edges, dilate, contours, aspect filter.
    Creates: edges image, cnts list, boxes list of (x, y, w, h).
    """
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
    """
    Function: ocr_plate
    Purpose: Read text from a binary region that looks like a plate.
    Methods: pytesseract.image_to_string with custom config.
    Creates: txt string.
    """
    txt = pytesseract.image_to_string(roi_bin, config=TESS_CFG)
    return txt.strip()


def normalize_plate(txt):
    """
    Function: normalize_plate
    Purpose: Clean up OCR text and unify common confusions.
    Methods: Upper-case, trim, remove spaces, O->0, I->1, Z->2, S->5, B->8.
    Creates: s working string, allowed character set string.
    """
    s = txt.upper().strip().replace(" ", "")
    s = s.replace("O", "0").replace("I", "1").replace("Z", "2")
    s = s.replace("S", "5").replace("B", "8")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(ch for ch in s if ch in allowed)

def pick_best_by_side(boxes, frame_width):
    """
    Function: pick_best_by_side
    Purpose: Select the best (largest-area) candidate independently
             for each side of the frame (left: exit, right: enter).
    Methods: Filter by x, sort by area, pick first per side.
    Creates: best_left, best_right tuples (x, y, w, h) or None.
    """
    left_zone = [b for b in boxes if b[0] < frame_width * EXIT_ZONE_X_LIMIT]
    right_zone = [b for b in boxes if b[0] > frame_width * ENTER_ZONE_X_LIMIT6]

    def area(b): return b[2] * b[3]
    left_zone.sort(key=area, reverse=True)
    right_zone.sort(key=area, reverse=True)

    best_left = left_zone[0] if left_zone else None
    best_right = right_zone[0] if right_zone else None
    return best_left, best_right

# -------- Server interaction ------------------------------------------------
def send_plate_event(plate, x, frame_width):
    """
    Function: send_plate_event
    Purpose: Call the server URL based on plate location (enter/exit).
    Methods: requests.get with timeout, status code handling.
    Creates: uri string, response object.
    """
    try:
        if x > frame_width * ENTER_ZONE_X_LIMIT:
            uri = f"{URL}/enter/{plate}"
        elif x < frame_width * EXIT_ZONE_X_LIMIT:
            uri = f"{URL}/exit/{plate}"
        else:
            return  # ignore center

        response = requests.get(uri, timeout=5)
        print(f"Sent {uri} -> Status: {response.status_code}")
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

# -------- Anti-spam helpers -------------------------------------------------
def start_block(plate):
    """
    Function: start_block
    Purpose: Begin a 10 s block window after sending 'plate'.
    Methods: Fill global vars with current monotonic time.
    Creates: block_active, block_deadline, last_* globals.
    """
    global block_active, block_deadline
    global last_sent_plate, last_seen_plate, last_seen_x
    global last_seen_time, last_seen_equal

    now = time.monotonic()
    last_sent_plate = plate
    block_active = True
    block_deadline = now + SIMILAR_TIMEOUT

    last_seen_plate = plate
    last_seen_x = 0
    last_seen_time = now
    last_seen_equal = True


def ocr_bbox(frame, box):
    """
    Function: ocr_bbox
    Purpose: Run OCR on a single bbox and normalize the text.
    Methods: crop, resize, preprocess_roi, ocr_plate, normalize_plate.
    Creates: roi, th, raw, plate strings.
    """
    (x, y, w, h) = box
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0,
                     interpolation=cv2.INTER_CUBIC)
    th = preprocess_roi(roi)
    raw = ocr_plate(th)
    plate = normalize_plate(raw)
    if PRINT_ALL_OCR:
        print(f"[side {'L' if x < frame.shape[1]*0.5 else 'R'}] "
              f"raw={raw!r} norm={plate!r}")
    return plate, x


def handle_candidate(candidate, x, frame_width):
    """
    Function: handle_candidate
    Purpose: Decide sending vs waiting for a 'candidate' plate.
    Methods: if/else using global block state; start_block().
    Creates: updates last_seen_*; calls send_plate_event when allowed.
    """
    global block_active, last_seen_plate, last_seen_x
    global last_seen_time, last_seen_equal, last_sent_plate

    now = time.monotonic()

    # Update what we currently see
    last_seen_plate = candidate
    last_seen_x = x
    last_seen_time = now
    last_seen_equal = (candidate == last_sent_plate)

    # If no block -> send immediately and start block
    if (not block_active) or (not last_sent_plate):
        send_plate_event(candidate, x, frame_width)
        start_block(candidate)
        return

    # In block: exact same plate -> wait (do not resend)
    if last_seen_equal:
        return

    # In block: similar/partial but different -> hold till timeout
    if is_partial_or_similar(candidate, last_sent_plate):
        return

    # In block: different enough -> send immediately and start new block
    send_plate_event(candidate, x, frame_width)
    start_block(candidate)


def check_timeout(frame_width):
    """
    Function: check_timeout
    Purpose: At/after deadline decide what to do and end the block.
    Methods: If still exact same -> print "please move on!";
             Else if similar but different -> send last_seen now.
    Creates: may call send_plate_event/start_block; may clear block.
    """
    global block_active, block_deadline, last_seen_time
    global last_seen_plate, last_seen_x, last_seen_equal
    global last_sent_plate

    if not block_active:
        return

    now = time.monotonic()
    if now < block_deadline:
        return

    # End the block window by decision:
    block_active = False

    # If we still see the exact same plate -> ask driver to move
    if last_seen_equal and ((now - last_seen_time) <= READ_PERIOD):
        if last_sent_plate:
            print(f"{last_sent_plate}, please move on!")
        return

    # If we see similar/partial but different -> send it now
    if last_seen_plate and \
       is_partial_or_similar(last_seen_plate, last_sent_plate) and \
       (last_seen_plate != last_sent_plate):
        send_plate_event(last_seen_plate, last_seen_x, frame_width)
        start_block(last_seen_plate)
        return

    # Otherwise nothing to do (block simply ends)
    # Next candidate will start a new block on send.


# ---- Main Loop -------------------------------------------------------------

def main():
    """
    Function: main
    Purpose: Open camera, run detection loop with 2 Hz OCR and
             simplified anti-spam, draw boxes/labels, handle keys.
    Methods: cv2.VideoCapture, find_plate_candidates, preprocess_roi,
             ocr_plate, normalize_plate, simple if/else state.
    Creates: cap, next_read_ts, labels, vis, key variables.
    """
    global next_read_ts

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        return
    print("Press 'q' to quit, 's' to save snapshot.")
    next_read_ts = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            boxes = find_plate_candidates(frame)
            best_left, best_right = pick_best_by_side(boxes, frame.shape[1])
            # Draw labels aligned to boxes
            labels = [""] * len(boxes)

            now = time.monotonic()
            if now >= next_read_ts:
                next_read_ts = now + READ_PERIOD

                best_left, best_right = pick_best_by_side(
                    boxes, frame.shape[1]
                )

                if best_left:
                    p, px = ocr_bbox(frame, best_left)
                    if 5 <= len(p) <= 8:
                        handle_candidate(p, px, frame.shape[1])
                        try:
                            i = boxes.index(best_left)
                            labels[i] = p
                        except ValueError:
                            pass

                if best_right:
                    p, px = ocr_bbox(frame, best_right)
                    if 5 <= len(p) <= 8:
                        handle_candidate(p, px, frame.shape[1])
                        try:
                            i = boxes.index(best_right)
                            labels[i] = p
                        except ValueError:
                            pass

                check_timeout(frame.shape[1])

            # Draw boxes/Labels every frame
            vis = frame.copy()
            for (x,y,w,h), lab in zip(boxes, labels) or [""] * len(boxes):
                cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0),2)
                if lab:
                    cv2.putText(vis, lab, (x,y - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,0.6,
                                (0,255,0), 2, cv2.LINE_AA)
                    
            cv2.imshow("Gate Watcher", vis)

            # Check for user input:
            key = cv2.waitKey(1) & 0xFF
            # This is tested in the frame window, not in the terminal

            if key == ord('q'):
                print("Exiting")
                break

            elif key == ord('s'):
                name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, vis)
                print(f"Frame saved into {name}")
            
    except KeyboardInterrupt:
        print("\nInterrupted byt Ctrl+C. Exiting...")

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
