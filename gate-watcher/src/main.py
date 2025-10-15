'''
GateWatcher v0.0.6 (20251015-1500):
The code monitors the gates of a parking lot.
If a license plate is detected on the right part of the creen,
It means a car is entering, so the code will issue a call to the server:
 URL/enter/<plate>
The server will reply with one of a following status_codes:
 210 - if the car was added correctly to the database.
 211 - if failed, e.g. car is already in the carpark.
If the plate is detected on the left part of the screen,
It means the car is exiting the parking lot, so the call will be:
 URL/exit/<plate>
The server will reply with one of the following status_codes:
 210 - if the fees for the parking were paid for this car
 211 - if the fees for the parking were NOT paid for this car
 212 - if any error occured
v0.0.3:
Antispam feature added:
 Read OCR every 500 milliseconds
 Block similar/partial re-sends for 10 seconds after last send
 If still the same number after timeout -> "{plate}, please move on"
 If after timeout the number is similar -> send the new plate
v0.0.5:
 - 'e' toggles ENTER-only scanning (right side).
 - 'x' toggles EXIT-only scanning (left side).
 - GPIO4 grounded: Entrance waiting
 - GPIO14 grounded: Exit waiting
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
import re


# --- NEW: GPIO + Sense HAT --------------------------------------------------
import RPi.GPIO as GPIO
from sense_hat import SenseHat

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
RE_PLATE = re.compile(r'^[A-Z]{3}\d{3}$')

# Allow sending on low confidence if we saw the same valid plate N times
LOW_CONF_OVERRIDE_ENABLED = True
LOW_CONF_SAME_COUNT = 3  # send even if avg_conf is low

# Tesseract configuration:
TESS_CFG = "--oem 1 --psm 7 -c " \
           "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_SAMPLE_CONF = 45.0   # Tesseract confidence threshold
MIN_FINAL_CONF  = 55.0   # Minimal average confidence to be sent
MIN_FINAL_LEN   = (6, 6) # Allowed length range for acceptable string
MIN_FINAL_SAMPLES = 2    # Need >=2 valid readings

# Image processing
AREA_MIN  = 1000
AREA_MAX  = 4000
AREA_STEP = 1000
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
# PREF_LEN_STRONG = {6, 7}       # strong length preference
# PREF_LEN_WEAK = {5, 8}         # weak length preference
PREF_LEN_STRONG = {6}          # strong length preference
PREF_LEN_WEAK = set()          # weak length preference

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
aggr_left  = {'samples': [], 'start_ts': 0.0, 'raw_counts': {}}
aggr_right = {'samples': [], 'start_ts': 0.0, 'raw_counts': {}}

show_zones = True  # press 'z' to toggle at runtime

# --- GPIO pins and LED arrow patterns ---------------------------------------
PIN_ENTER = 14   # BCM 14 -> Enter (Up arrow)
PIN_EXIT  = 4    # BCM 4  -> Exit  (Down arrow)
DEBOUNCE_SEC = 0.05

RED  = (255, 0,   0)
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
    Purpose: Add a sample into the bucket; always bump raw frequency,
    even when confidence is low. Keep 'samples' filtered by conf.
    Methods: Set start_ts on the first sample; bump raw_counts[plate];
    append to 'samples' only if conf >= MIN_SAMPLE_CONF.
    Creates: tuples in 'samples': (plate, conf, x, ts); updates dict.
    """
    now = time.monotonic()
    if not bucket['samples'] and not bucket['raw_counts']:
        bucket['start_ts'] = now
    # Bump raw frequency for normalized plate (non-empty only)
    if plate:
        cnt = bucket['raw_counts'].get(plate, 0)
        bucket['raw_counts'][plate] = cnt + 1
    # Keep confidence-filtered list for normal scoring
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
        if not plate:
            # Ignore empty reads; they do not help selection
            continue
        s = stats.setdefault(plate, {'sum': 0.0, 'n': 0, 'x': x})
        s['sum'] += conf
        s['n'] += 1
        s['x'] = x  # use the most recent x
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


def is_valid_plate(plate: str) -> bool:
    """
    Function: is_valid_plate
    Purpose: Enforce exact AAA999 pattern before any send.
    Methods: Regex match against RE_PLATE.
    Creates: None.
    """
    return bool(RE_PLATE.fullmatch(plate))


def maybe_finalize(bucket, frame_width):
    """
    Function: maybe_finalize
    Purpose: Finalize when enough samples or window passed; apply
    strict filters; allow low-confidence override on repetition.
    Methods: Count/window check; choose_best; strict AAA999 filters;
    fallback: raw_counts >= LOW_CONF_SAME_COUNT -> send anyway.
    Creates: Calls handle_candidate(); clears bucket afterwards.
    """
    if not bucket['samples'] and not bucket['raw_counts']:
        return
    now = time.monotonic()
    enough_count = (len(bucket['samples']) >= AGGR_MAX_SAMPLES)
    enough_time = ((now - bucket['start_ts']) >= AGGR_WINDOW)
    if not (enough_count or enough_time):
        return

    # Normal path: score by confidence-filtered samples
    plate, x, avg_conf, n = ("", 0, 0.0, 0)
    if bucket['samples']:
        plate, x, avg_conf, n = choose_best(bucket['samples'])

    # Decide by strict filters first
    def strict_ok(p, avg, nn):
        if not p:
            return False
        if not (MIN_FINAL_LEN[0] <= len(p) <= MIN_FINAL_LEN[1]):
            return False
        letters = sum(ch.isalpha() for ch in p)
        digits = sum(ch.isdigit() for ch in p)
        if not (letters == 3 and digits == 3):  # exact AAA999
            return False
        if (avg < MIN_FINAL_CONF) or (nn < MIN_FINAL_SAMPLES):
            return False
        return True

    # If strict path succeeds -> send; else try low-conf override
    if strict_ok(plate, avg_conf, n):
        clear_aggr(bucket)
        handle_candidate(plate, x, frame_width)
        return

    # ---- Low-confidence override based on raw repetition ----
    if LOW_CONF_OVERRIDE_ENABLED:
        # Pick the most frequent valid AAA999 in raw_counts
        best_raw_plate, best_raw_count = "", 0
        best_raw_x = x  # reuse latest x if unknown
        for p, c in bucket['raw_counts'].items():
            if c > best_raw_count and is_valid_plate(p):
                best_raw_plate, best_raw_count = p, c
        if best_raw_plate and best_raw_count >= LOW_CONF_SAME_COUNT:
            clear_aggr(bucket)
            handle_candidate(best_raw_plate, best_raw_x, frame_width)
            return

    # Nothing qualified -> clear and do nothing
    clear_aggr(bucket)


# ----------- OCR on bbox (with confidence) ---------------------------------
def ocr_bbox(frame, box):
    """
    Function: ocr_bbox
    Purpose: Run OCR on a single bbox and normalize the text; also
    compute OCR confidence and print side based on zone thresholds.
    Methods: crop, resize, preprocess_roi, ocr_text_and_conf, normalize;
    side='L' if x in EXIT zone, 'R' if in ENTER zone, else 'C'.
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
        fw = frame.shape[1]
        if x < fw * EXIT_ZONE_X_LIMIT:
            side = 'L'   # Exit zone (left)
        elif x > fw * ENTER_ZONE_X_LIMIT:
            side = 'R'   # Enter zone (right)
        else:
            side = 'C'   # Middle (ignored)
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
    Purpose: Call the server URL based on plate location (enter/exit),
             and interpret recycled status codes according to operation.
    Methods: Decide op by x; requests.get; branch by op and status code.
    Creates: 'op' string; 'uri' string; prints human-friendly messages.
    """
    try:
        # Decide operation by horizontal position
        print(f"The plate location is {x}")
        if x > frame_width * ENTER_ZONE_X_LIMIT:
            op = 'enter'
            uri = f"{URL}/enter/{plate}"
        elif x < frame_width * EXIT_ZONE_X_LIMIT:
            op = 'exit'
            uri = f"{URL}/exit/{plate}"
        else:
            return  # ignore center (no-op)

        response = requests.get(uri, timeout=5)
        code = response.status_code
        print(f"Sent {uri} -> HTTP {code}")

        # Interpret recycled codes per operation
        if op == 'enter':
            # (210: added | 211: already exists | 213: error or invalid)
            if code == 210:
                print("Enter: plate added, open gate.")
                # servo.set_gate(0, False)  # optionally open entry
            elif code == 211:
                print("Enter: already exists, do not open gate.")
            elif code == 213:
                print("Enter: invalid plate format or server error.")
            else:
                print(f"Enter: unexpected status {code}")

        else:  # op == 'exit'
            # (210: paid, exit allowed | 211: not paid | 212: not found | 213: error)
            if code == 210:
                print("Exit: paid up, exit permitted, open gate.")
                # servo.set_gate(1, False)  # optionally open exit
            elif code == 211:
                print("Exit: NOT paid, keep gate closed.")
            elif code == 212:
                print("Exit: plate not found.")
            elif code == 213:
                print("Exit: server error or invalid plate.")
            else:
                print(f"Exit: unexpected status {code}")

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
    Purpose: Decide what to do at/after anti-spam deadline.
             - If the SAME plate is still present, do NOT resend; show
               "please move on!" and RE-ARM the block for another
               SIMILAR_TIMEOUT window.
             - If a similar/partial BUT DIFFERENT plate is present,
               send it now and start a new block.
             - Otherwise, simply end the block.
    Methods: Compare 'now' vs 'block_deadline'; inspect last_seen_* and
             last_sent_plate; call send_plate_event/start_block when needed.
    Creates: May keep 'block_active' True (re-arm) or set it False;
             may emit one send when a new similar plate showed up.
    """
    global block_active, block_deadline, last_seen_time
    global last_seen_plate, last_seen_x, last_seen_equal
    global last_sent_plate
    if not block_active:
        return
    now = time.monotonic()
    if now < block_deadline:
        return

    # === We are at/after the deadline ===
    # Case 1: The exact same plate is still visible -> suppress resend,
    # show message, and re-arm the block for another timeout window.
    if last_seen_equal and ((now - last_seen_time) <= READ_PERIOD):
        if last_sent_plate:
            print(f"{last_sent_plate}, please move on!")
        # Re-arm the anti-spam block instead of ending it.
        block_deadline = now + SIMILAR_TIMEOUT
        block_active = True
        return

    # Case 2: A similar/partial but different plate is visible -> send now
    if (last_seen_plate and
        is_partial_or_similar(last_seen_plate, last_sent_plate) and
        (last_seen_plate != last_sent_plate)):
        send_plate_event(last_seen_plate, last_seen_x, frame_width)
        start_block(last_seen_plate)
        return

    # Case 3: Nothing relevant -> end the block
    block_active = False

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
    bucket['raw_counts'].clear()


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
    print("GPIO4=Exit(blue v), GPIO14=Enter(red ^); both -> EXIT prioritized.")

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
