'''
GateWatcher v0.0.5:
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

v0.0.3:
Antispam feature added:
    Read OCR every 500 milliseconds
    Block similar/partial re-sends for 10 seconds after last send
    If still the same number after timeout - print "{plate}, please move on"
    If after timeout the number is similar - send the new plate

v0.0.5:

 - 'e' toggles ENTER-only scanning (right side).
 - 'x' toggles EXIT-only scanning (left side).
 - Multi-sample OCR filter picks the best plate over a short window.
'''
import requests
import servo # servo.py
from time import sleep
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import time
from datetime import datetime

def testServerConnection():
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

# Replace with your target IP (and include http:// or https://)
url = "http://127.0.0.1:5000"

CAMERA_INDEX = 0                 # Ususaly '0' for the first connected camera
WEB_PI_IP = "http://127.0.0.1"  # Holds the IP of the web server pi
URL = f"{WEB_PI_IP}:5000"        # Holds the full address of the server

MIN_AREA = 2000              # 
ASPECT_MIN = 2.0             # 
ASPECT_MAX = 6.0             # 
MAX_CANDIDATES = 6           # 
PRINT_ALL_OCR = True         # 

# Tesseract configuration:
TESS_CFG = "--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# OCR constants:
EXIT_ZONE_X_LIMIT  = 0.4     # Left  side -> Exit
ENTER_ZONE_X_LIMIT = 0.6     # Right side -> Entrance

# Anti-Spam controls:
READ_PERIOD = 0.5            # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0       # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1     # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1   # Allow one missing char in a prefix match

READ_PERIOD = 0.5           # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0      # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1    # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1  # Allow one missing char in a prefix match

# Multi-OCR aggregation controls:
AGGR_MAX_SAMPLES = 5           # collect up to N OCR samples
AGGR_WINDOW = 0.7              # or until this many seconds pass
PREF_LEN_STRONG = {6, 7}       # strong length preference
PREF_LEN_WEAK = {5, 8}         # weak length preference

# Global Anti-spam state variables:
last_sent_plate = ""        # last plate sent to server
block_active = False        # are we in 10 s block window?
block_deadline = 0.0        # monotonic deadline when block expires

last_seen_plate = ""        # most recently observed candidate
last_seen_x = 0             # x coordinate of last seen candidate
last_seen_time = 0.0        # monotonic time when we saw last candidate
last_seen_equal = False     # whether last seen equals last_sent_plate

next_read_ts = 0.0          # throttle OCR (monotonic time)

#Scan mode controlled by keys/buttons
scan_mode = 'idle'

#OCR aggregation "buckets"
aggr_left  = {'samples': [], 'start_ts': 0.0}
aggr_right = {'samples': [], 'start_ts': 0.0}

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


def ocr_text_and_conf(img_bin):
    """
    Function: ocr_text_and_conf
    Purpose: Run Tesseract OCR and return text with avg confidence.
    Methods: pytesseract.image_to_data with Output.DICT; average conf.
    Creates: data dict, confs list, avg_conf float, raw string.
    """
    data = pytesseract.image_to_data(
        img_bin, config=TESS_CFG, output_type=Output.DICT
    )
    confs = []
    for c in data.get('conf', []):
        try:
            v = int(c)
            if v >= 0:
                confs.append(v)
        except ValueError:
            continue
    avg_conf = float(sum(confs)) / max(len(confs), 1)
    raw = " ".join([w for w in data.get('text', []) if w.strip()])
    return raw, avg_conf


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
    right_zone = [b for b in boxes if b[0] > frame_width * ENTER_ZONE_X_LIMIT]

    def area(b): return b[2] * b[3]
    left_zone.sort(key=area, reverse=True)
    right_zone.sort(key=area, reverse=True)

    best_left = left_zone[0] if left_zone else None
    best_right = right_zone[0] if right_zone else None
    return best_left, best_right


# ------------- OCR on bbox (with confidence) -------------------------------
def ocr_bbox(frame, box):
    """
    Function: ocr_bbox
    Purpose: Run OCR on a single bbox and normalize the text; also
             compute OCR confidence.
    Methods: crop, resize, preprocess_roi, ocr_text_and_conf, normalize.
    Creates: roi, th, raw, plate strings; conf float; px int.
    """
    (x, y, w, h) = box
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(
        roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
    )
    th = preprocess_roi(roi)
    raw, conf = ocr_text_and_conf(th)
    plate = normalize_plate(raw)
    if PRINT_ALL_OCR:
        side = 'L' if x < frame.shape[1] * 0.5 else 'R'
        print(
            f"[side {side}] raw={raw!r} norm={plate!r} conf={conf:.1f}"
        )
    return plate, conf, x


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


# ---------------- Multi-OCR aggregation ------------------------------------
def clear_aggr(bucket):
    """
    Function: clear_aggr
    Purpose: Reset aggregation bucket to empty state.
    Methods: Clear list and zero start_ts.
    Creates: empties 'samples', sets 'start_ts' to 0.
    """
    bucket['samples'].clear()
    bucket['start_ts'] = 0.0


def add_sample(bucket, plate, conf, x):
    """
    Function: add_sample
    Purpose: Add an OCR sample to the aggregation bucket.
    Methods: Append tuple; set start_ts on the first sample.
    Creates: tuples in 'samples': (plate, conf, x, ts).
    """
    now = time.monotonic()
    if not bucket['samples']:
        bucket['start_ts'] = now
    bucket['samples'].append((plate, float(conf), int(x), now))


def choose_best(samples):
    """
    Function: choose_best
    Purpose: From aggregated samples choose the best plate string.
    Methods: Group by plate; score=avg_conf + 10*freq + length bonus.
    Creates: dict per plate with freq, avg_conf, last_x; returns tuple.
    """
    if not samples:
        return "", 0
    stats = {}
    for plate, conf, x, ts in samples:
        if not plate:
            # Ignore empty reads; they do not help selection
            continue
        s = stats.setdefault(plate, {'sum': 0.0, 'n': 0, 'x': x})
        s['sum'] += conf
        s['n'] += 1
        s['x'] = x  # use the most recent x
    if not stats:
        return "", 0
    best_plate, best_score, best_x = "", -1e9, 0
    for p, s in stats.items():
        avg_conf = s['sum'] / max(s['n'], 1)
        freq = s['n']
        L = len(p)
        len_bonus = 0.0
        if L in PREF_LEN_STRONG:
            len_bonus = 5.0
        elif L in PREF_LEN_WEAK:
            len_bonus = 2.5
        score = avg_conf + 10.0 * freq + len_bonus
        if score > best_score:
            best_plate = p
            best_score = score
            best_x = s['x']
    return best_plate, best_x


def maybe_finalize(bucket, frame_width):
    """
    Function: maybe_finalize
    Purpose: Decide when to stop sampling and emit the best candidate.
    Methods: Finalize on count>=AGGR_MAX_SAMPLES or time window passed.
    Creates: Calls handle_candidate(); clears bucket afterwards.
    """
    if not bucket['samples']:
        return
    now = time.monotonic()
    enough_count = (len(bucket['samples']) >= AGGR_MAX_SAMPLES)
    enough_time = ((now - bucket['start_ts']) >= AGGR_WINDOW)
    if not (enough_count or enough_time):
        return
    plate, x = choose_best(bucket['samples'])
    clear_aggr(bucket)
    if 5 <= len(plate) <= 8:
        handle_candidate(plate, x, frame_width)


# ---------------- Scan mode handling ---------------------------------------
def toggle_mode(new_mode):
    """
    Function: toggle_mode
    Purpose: Toggle scanning mode between 'idle', 'enter', and 'exit'.
    Methods: Switch logic; clearing aggregators on mode change.
    Creates: updates global scan_mode; resets aggr buckets.
    """
    global scan_mode
    global aggr_left, aggr_right
    if scan_mode == new_mode:
        scan_mode = 'idle'
    else:
        scan_mode = new_mode
    clear_aggr(aggr_left)
    clear_aggr(aggr_right)
    print(f"Scan mode: {scan_mode.upper() if scan_mode!='idle' else 'IDLE'}")


# ---------------- Main Loop -------------------------------------------------
def main():
    """
    Function: main
    Purpose: Open camera, run detection loop with 2 Hz OCR and
             anti-spam; react to 'e'/'x' modes and multi-OCR filter.
    Methods: cv2.VideoCapture, find_plate_candidates, ocr_bbox,
             aggregation, anti-spam state machine.
    Creates: cap, next_read_ts, labels, vis, key variables.
    """
    global next_read_ts, scan_mode
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        return
    print("Press 'e' ENTER-only, 'x' EXIT-only, 's' snapshot, 'q' quit.")
    next_read_ts = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            boxes = find_plate_candidates(frame)
            best_left, best_right = pick_best_by_side(
                boxes, frame.shape[1]
            )

            # Throttled OCR sampling in active modes only
            now = time.monotonic()
            if now >= next_read_ts:
                next_read_ts = now + READ_PERIOD

                if scan_mode == 'exit' and best_left:
                    p, conf, px = ocr_bbox(frame, best_left)
                    if p:
                        add_sample(aggr_left, p, conf, px)
                    maybe_finalize(aggr_left, frame.shape[1])

                if scan_mode == 'enter' and best_right:
                    p, conf, px = ocr_bbox(frame, best_right)
                    if p:
                        add_sample(aggr_right, p, conf, px)
                    maybe_finalize(aggr_right, frame.shape[1])

                check_timeout(frame.shape[1])

            # Draw boxes/labels every frame
            vis = frame.copy()
            for (x, y, w, h) in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # HUD: show current scan mode
            hud = "MODE: "
            if scan_mode == 'idle':
                hud += "IDLE"
                color = (200, 200, 200)
            elif scan_mode == 'enter':
                hud += "ENTER (Right)"
                color = (0, 255, 0)
            else:
                hud += "EXIT (Left)"
                color = (0, 255, 255)
            cv2.putText(
                vis, hud, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )

            cv2.imshow("Gate Watcher", vis)

            # Check for user input (window has the focus)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Exiting")
                break
            elif key == ord('s'):
                name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, vis)
                print(f"Frame saved into {name}")
            elif key == ord('e'):   # ENTER side only
                toggle_mode('enter')
            elif key == ord('x'):   # EXIT side only
                toggle_mode('exit')

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
