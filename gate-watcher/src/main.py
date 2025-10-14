'''
GateWatcher v0.0.6 (20251015-1127):
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
 If still the same number after timeout -> "{plate}, please move on"
 If after timeout the number is similar -> send the new plate
v0.0.5:
 - 'e' toggles ENTER-only scanning (right side).
 - 'x' toggles EXIT-only scanning (left side).
 - Multi-sample OCR filter picks the best plate over a short window.
'''
import requests
import servo  # servo.py
from time import sleep
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import time
from datetime import datetime

# --- NEW: GPIO + Sense HAT --------------------------------------------------
import RPi.GPIO as GPIO
from sense_hat import SenseHat

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
# servo.set_gate(0, False) # Open gate 0 (Entry gate)
# servo.set_gate(1, False) # Open gate 1 (Exit gate)
# sleep(1)
# Close Gates
# servo.set_gate(0, True)  # Close gate 0 (Entry gate)
# servo.set_gate(1, True)  # Close gate 1 (Exit gate)

# ---- Configuration ---------------------------------------------------------
# Replace with your target IP (and include http:// or https://)
url = "http://127.0.0.1:5000"
CAMERA_INDEX = 0  # Ususaly '0' for the first connected camera
WEB_PI_IP = "http://127.0.0.1"  # Holds the IP of the web server pi
URL = f"{WEB_PI_IP}:5000"  # Holds the full address of the server
ASPECT_MIN = 2.0 #
ASPECT_MAX = 6.0 #
MAX_CANDIDATES = 10 #
PRINT_ALL_OCR = True #


# OCR constants:
EXIT_ZONE_X_LIMIT = 0.52  # Left side -> Exit
ENTER_ZONE_X_LIMIT = 0.58 # Right side -> Entrance

# Tesseract configuration:
TESS_CFG = "--oem 1 --psm 7 -c " \
           "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_SAMPLE_CONF = 45.0   # Tesseract confidence threshold
MIN_FINAL_CONF  = 55.0   # Minimal average confidence to be sent
MIN_FINAL_LEN   = (5, 8) # Allowed length range for acceptable string
MIN_FINAL_SAMPLES = 2    # Need >=2 valid readings

# Image processing
AREA_MIN  = 1000
AREA_MAX  = 10000
AREA_STEP = 200
AREA_ABS_MIN = 200
area_min  = AREA_MIN
area_max  = AREA_MAX

# Camera Constants
CAMERA_RESOLUTION = (1280, 720)  # or (1920, 1080), (640, 480)

# Anti-Spam controls:
READ_PERIOD = 0.5              # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0         # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1       # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1     # Allow one missing char in a prefix match
READ_PERIOD = 0.5              # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0         # seconds to block similar/partial plates
SIMILAR_DISTANCE_MAX = 1       # Levenshtein distance threshold
PARTIAL_MIN_MATCH_DROP = 1     # Allow one missing char in a prefix match

# Multi-OCR aggregation controls:
AGGR_MAX_SAMPLES = 5           # collect up to N OCR samples
AGGR_WINDOW = 0.7              # or until this many seconds pass
PREF_LEN_STRONG = {6, 7}       # strong length preference
PREF_LEN_WEAK = {5, 8}         # weak length preference

# HW Constants
GPIO_STABLE_MS = 300           # 300 mSec jitter prevention

# Global Anti-spam state variables:
last_sent_plate = ""           # last plate sent to server
block_active = False           # are we in 10 s block window?
block_deadline = 0.0           # monotonic deadline when block expires
last_seen_plate = ""           # most recently observed candidate
last_seen_x = 0                # x coordinate of last seen candidate
last_seen_time = 0.0           # monotonic time when we saw last candidate
last_seen_equal = False        # whether last seen equals last_sent_plate
next_read_ts = 0.0             # throttle OCR (monotonic time)

# Scan mode controlled by keys/buttons
scan_mode = 'idle'

# OCR aggregation "buckets"
aggr_left  = {'samples': [], 'start_ts': 0.0}
aggr_right = {'samples': [], 'start_ts': 0.0}

show_zones = True  # press 'z' to toggle at runtime

# --- GPIO pins and LED arrow patterns ---------------------------------------
PIN_ENTER = 4    # BCM 4  -> Enter (Up arrow)
PIN_EXIT  = 14   # BCM 14 -> Exit  (Down arrow)
DEBOUNCE_SEC = 0.05

RED  = (255, 0, 0)
BLUE = (0,   0, 255)
BLK  = (0,   0,   0)

UP_RED = [
    "00000000",
    "00r00000",
    "0rrr0000",
    "00r00000",
    "00r00000",
    "00r00000",
    "00r00000",
    "00000000",
]

DOWN_BLUE = [
    "00000000",
    "00000b00",
    "00000b00",
    "00000b00",
    "00000b00",
    "0000bbb0",
    "00000b00",
    "00000000",
]

BOTH = [
    "00000000",
    "00r00b00",
    "0rrr0b00",
    "00r00b00",
    "00r00b00",
    "00r0bbb0",
    "00r00b00",
    "00000000",
]

# ---- Functions -------------------------------------------------------------
# -------- GPIO handling -----------------------------------------------------
def pattern_to_pixels(pat):
    """
    Function: pattern_to_pixels
    Purpose: Convert 8x8 char pattern into a list of 64 RGB tuples.
    Methods: Iterate rows/cols; map 'r'->RED, 'b'->BLUE, otherwise BLK.
    Creates: 'pix' list of tuples (R,G,B).
    """
    pix = []
    for row in pat:
        for ch in row:
            if ch == 'r':
                pix.append(RED)
            elif ch == 'b':
                pix.append(BLUE)
            else:
                pix.append(BLK)
    return pix

def show_arrows(sense, enter_low, exit_low):
    """
    Function: show_arrows
    Purpose: Display arrows on Sense HAT based on GPIO state.
    Methods: Choose UP/DOWN/BOTH; call sense.set_pixels(); clear if none.
    Creates: local 'pixels' list when any input is active.
    """
    if enter_low and exit_low:
        sense.set_pixels(pattern_to_pixels(BOTH))
    elif exit_low:
        sense.set_pixels(pattern_to_pixels(DOWN_BLUE))
    elif enter_low:
        sense.set_pixels(pattern_to_pixels(UP_RED))
    else:
        sense.clear()

def read_gpio_state():
    """
    Function: read_gpio_state
    Purpose: Sample GPIO pins and return booleans (enter_low, exit_low).
    Methods: GPIO.input() with pull-ups; LOW means grounded/active.
    Creates: two bools.
    """
    enter_low = (GPIO.input(PIN_ENTER) == GPIO.LOW)
    exit_low  = (GPIO.input(PIN_EXIT)  == GPIO.LOW)
    return enter_low, exit_low

def force_mode(new_mode):
    """
    Function: force_mode
    Purpose: Deterministically set scan_mode, clear aggregators on change.
    Methods: Assign 'scan_mode' and reset aggr_left/aggr_right.
    Creates: updates global scan_mode; prints mode banner.
    """
    global scan_mode, aggr_left, aggr_right
    if scan_mode != new_mode:
        scan_mode = new_mode
        clear_aggr(aggr_left)
        clear_aggr(aggr_right)
        print(f"Scan mode: "
              f"{scan_mode.upper() if scan_mode!='idle' else 'IDLE'}")

# --------------------------- Similarity -------------------------------------
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
                dp[i - 1][j] + 1,         # deletion
                dp[i][j - 1] + 1,         # insertion
                dp[i - 1][j - 1] + cost   # substitution
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

# ---------------- Image preprocessing and OCR -------------------------------
import subprocess


def set_brightness(value):
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0", "--set-ctrl", f"brightness={value}"
    ])
    print(f"Яркость: {value}")
    return value

def set_contrast(value):
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0", "--set-ctrl", f"contrast={value}"
    ])
    print(f"Контраст: {value}")
    return value

def set_gain(value):
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0", "--set-ctrl", f"gain={value}"
    ])
    print(f"Gain: {value}")
    return value


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
    _, th = cv2.threshold(
        clahe.apply(filt), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY
    )
    return th


def find_plate_candidates(frame_bgr):
    """
    Function: find_plate_candidates
    Purpose: Detect rectangular regions that may contain a plate.
    Methods: Gray, blur, Canny edges, dilate, contours, aspect filter,
             binary mask for black-on-white text.
    Creates: edges image, cnts list, boxes list of (x, y, w, h).
    """
    # 50 shades of Grey
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Blurring to remove the noise
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Implementing binary mask: Search for black-on-white text
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Combining the mask with the original image
    combined = cv2.bitwise_and(gray, binary_mask)

    # highlighting the boundaries
    edges = cv2.Canny(combined, 80, 200)

    # Expanding boudaries
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)

    # Searching for contours
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < area_min or area > area_max:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / max(h, 1)
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            boxes.append((x, y, w, h))

    # Sorting by area, returning the best candidates
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
    Purpose: Clean up OCR text with minimal ambiguity fixes.
    Methods: Upper-case, trim, remove spaces; O->0 and I->1 selectively.
    Creates: s working string, allowed character set string.
    """
    s = txt.upper().strip().replace(" ", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(ch for ch in s if ch in allowed)

    # Лёгкая замена: только O->0 и I->1 если окружены цифрами
    def smart_swap(chars):
        out = []
        for i, ch in enumerate(chars):
            if ch in ("O", "I"):
                left_d  = (i > 0 and chars[i-1].isdigit())
                right_d = (i+1 < len(chars) and chars[i+1].isdigit())
                if left_d or right_d:
                    ch = "0" if ch == "O" else "1"
            out.append(ch)
        return "".join(out)

    return smart_swap(s)


def pick_best_by_side(boxes, frame_width):
    """
    Function: pick_best_by_side
    Purpose: Select the best (largest-area) candidate independently
             for each side of the frame (left: exit, right: enter).
    Methods: Filter by x, sort by area, pick first per side.
    Creates: best_left, best_right tuples (x, y, w, h) or None.
    """
    left_zone = [b for b in boxes
                 if b[0] < frame_width * EXIT_ZONE_X_LIMIT]
    right_zone = [b for b in boxes
                  if b[0] > frame_width * ENTER_ZONE_X_LIMIT]

    def area(b): return b[2] * b[3]

    left_zone.sort(key=area, reverse=True)
    right_zone.sort(key=area, reverse=True)

    best_left = left_zone[0] if left_zone else None
    best_right = right_zone[0] if right_zone else None
    return best_left, best_right


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
    # Отбрасываем пустые строки и сэмплы с низкой уверенностью
    if (not plate) or (float(conf) < MIN_SAMPLE_CONF):
        return
    bucket['samples'].append((plate, float(conf), int(x), now))


def choose_best(samples):
    """
    Function: choose_best
    Purpose: From aggregated samples choose the best plate string.
    Methods: Group by plate; score=avg_conf + 10*freq + length bonus.
    Creates: dict per plate with freq, avg_conf, last_x; returns tuple.
    """
    if not samples:
        return "", 0, 0.0, 0
    stats = {}
    for plate, conf, x, ts in samples:
        s = stats.setdefault(plate, {'sum': 0.0, 'n': 0, 'x': x})
        s['sum'] += conf
        s['n'] += 1
        s['x'] = x  # последний x
    if not stats:
        return "", 0, 0.0, 0

    best_plate, best_score, best_x = "", -1e9, 0
    best_avg, best_n = 0.0, 0
    for p, s in stats.items():
        avg_conf = s['sum'] / max(s['n'], 1)
        freq = s['n']
        L = len(p)
        len_bonus = 5.0 if L in PREF_LEN_STRONG else \
                    (2.5 if L in PREF_LEN_WEAK else 0.0)
        score = avg_conf + 10.0 * freq + len_bonus
        if score > best_score:
            best_plate = p
            best_score = score
            best_x = s['x']
            best_avg = avg_conf
            best_n = freq
    return best_plate, best_x, best_avg, best_n


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
    enough_time  = ((now - bucket['start_ts']) >= AGGR_WINDOW)
    if not (enough_count or enough_time):
        return

    plate, x, avg_conf, n = choose_best(bucket['samples'])
    clear_aggr(bucket)

    # Финальные фильтры: длина, уверенность, число валидных чтений,
    # грубая проверка состава (минимум 2 буквы и 2 цифры)
    if not plate:
        return
    if not (MIN_FINAL_LEN[0] <= len(plate) <= MIN_FINAL_LEN[1]):
        return
    letters = sum(ch.isalpha() for ch in plate)
    digits  = sum(ch.isdigit() for ch in plate)
    if (letters < 2) or (digits < 2):
        return
    if (avg_conf < MIN_FINAL_CONF) or (n < MIN_FINAL_SAMPLES):
        return

    handle_candidate(plate, x, frame_width)



# ----------- OCR on bbox (with confidence) ---------------------------------
def ocr_bbox(frame, box):
    """
    Function: ocr_bbox
    Purpose: Run OCR on a single bbox and normalize the text; also
             compute OCR confidence.
    Methods: crop, resize, preprocess_roi, ocr_text_and_conf, normalize.
    Creates: plate strings; conf float; px int.
    """
    (x, y, w, h) = box
    roi = frame[y:y + h, x:x + w]
    roi = cv2.resize(roi, None, fx=2.0, fy=2.0,
                     interpolation=cv2.INTER_CUBIC)
    th = preprocess_roi(roi)
    raw, conf = ocr_text_and_conf(th)
    plate = normalize_plate(raw)
    if PRINT_ALL_OCR:
        side = 'L' if x < frame.shape[1] * 0.5 else 'R'
        print(f"[side {side}] raw={raw!r} norm={plate!r} conf={conf:.1f}")

    return plate, conf, x



def draw_zones(vis):
    """
    Function: draw_zones
    Purpose: Draw EXIT/ENTER vertical borders and shaded regions.
    Methods: Compute pixel x from relative limits; draw lines and overlays.
    Creates: no state, draws on 'vis' in-place.
    """
    h, w = vis.shape[:2]
    x_exit = int(w * EXIT_ZONE_X_LIMIT)
    x_enter = int(w * ENTER_ZONE_X_LIMIT)

    # semi-transparent shading for zones
    overlay = vis.copy()
    # left (exit) zone in cyan-ish
    cv2.rectangle(overlay, (0, 0), (x_exit, h), (255, 255, 0), -1)
    # right (enter) zone in green-ish
    cv2.rectangle(overlay, (x_enter, 0), (w, h), (0, 255, 0), -1)
    # blend
    alpha = 0.15
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # draw the vertical boundary lines
    cv2.line(vis, (x_exit, 0), (x_exit, h), (0, 255, 255), 2)
    cv2.line(vis, (x_enter, 0), (x_enter, h), (0, 255, 0), 2)

    # labels with pixel coordinates
    cv2.putText(
        vis, f"EXIT < x<={x_exit}px", (10, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, f"ENTER >= {x_enter}px", (10, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )


# -------------------- Server interaction -----------------------------------
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

# --------------------- Anti-spam helpers -----------------------------------
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

# -------------------- Multi-OCR aggregation --------------------------------
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

# -------------------- Scan mode handling -----------------------------------
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

# ----------------------------- Main Loop -----------------------------------
def main():
    """
    Function: main
    Purpose: Open camera, run detection loop with 2 Hz OCR and
             anti-spam; react to GPIO14/GPIO4 and multi-OCR filter.
    Methods: cv2.VideoCapture, find_plate_candidates, ocr_bbox,
             aggregation, anti-spam state machine; Sense HAT display.
    Creates: cap, next_read_ts, labels, vis, key variables.
    """
    global next_read_ts, scan_mode, show_zones

    # Init Sense HAT and GPIO
    sense = SenseHat()
    sense.clear()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_ENTER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(PIN_EXIT,  GPIO.IN, pull_up_down=GPIO.PUD_UP)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        GPIO.cleanup()
        sense.clear()
        return

    print("Press 'e' ENTER-only, 'x' EXIT-only, 's' snapshot, 'z' toggle zones, 'q' quit.")
    print("GPIO14=Exit(red ^), GPIO4=Enter(blue v); both -> EXIT prioritized.")

    next_read_ts = 0.0
    prev_state = (None, None)

    # Camera picture settings
    brightness = 128
    contrast = 128
    gain = 128

    # OCR area settings
    global area_min, area_max

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Read GPIO and show arrows; set mode with EXIT priority
            enter_low, exit_low = read_gpio_state()
            show_arrows(sense, enter_low, exit_low)
            if exit_low:
                force_mode('exit')
            elif enter_low:
                force_mode('enter')
            else:
                force_mode('idle')

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
            if show_zones:
                draw_zones(vis)
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
            elif key == ord('e'):  # ENTER side only
                toggle_mode('enter')
            elif key == ord('x'):  # EXIT side only
                toggle_mode('exit')
            elif key == ord('z'):    # Toggle enter/exit zones
                show_zones = not show_zones
                print(f"Zone overlay: {'ON' if show_zones else 'OFF'}")
            elif key == ord(','):
                brightness -= 5
                brightness = set_brightness(brightness)

            elif key == ord('.'):
                brightness += 5
                brightness = set_brightness(brightness)

            elif key == ord(';'):
                contrast -= 5
                contrast = set_contrast(contrast)

            elif key == ord("'"):
                contrast += 5
                contrast = set_contrast(contrast)
            
            elif key == ord('['):
                area_min = max(AREA_ABS_MIN, area_min - AREA_STEP)
                area_max = max(area_min + 2.5*AREA_ABS_MIN, area_max - AREA_STEP)
                print(f"Area range: {area_min}–{area_max}")

            elif key == ord(']'):
                area_min = min(60000, area_min + AREA_STEP)
                area_max = min(100000, area_max + AREA_STEP)
                print(f"Area range: {area_min}–{area_max}")

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        sense.clear()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
