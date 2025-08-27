'''
GateWatcher v0.0.8 (20251018-1250):
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
v0.0.7, Spring cleaning:
Removed:
 - levenshtein(...)
 - is_partial_or_similar
 - SIMILAR_DISTANCE_MAX
 - PARTIAL_MIN_MATCH_DROP
 - block_active, block_deadline, last_seen_plate, last_seen_x, last_seen_time, last_seen_equal
 - start_block(...)
 - check_timeout(...)
 - LOW_CONF_OVERRIDE_ENABLED
 - LOW_CONF_SAME_COUNT
 - raw_counts
 - len_bonus calculation
 - PREF_LEN_STRONG
 - PREF_LEN_WEAK 

Added:
 - BLOCK_TIMEOUT

Changed:
 - aggr_left + aggr_right
 - clear_aggr(...)
 - add_sample(...)
 - choose_best(...)
 - maybe_finalize(...)
 - handle_candidate(...)
 - main(...):
 -- check_timeout(...)
v0.0.8: Removing piHat
'''
import requests
# import servo  # servo.py
from time import sleep
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import time
from datetime import datetime
import re
import RPi.GPIO as GPIO

# Needed for Servo, LEDs, and Buttons
GPIO.setmode(GPIO.BCM)
'''
Entry pinout
13 = red
6 = green
5 = blue
19 = entry button

Exit pintout
9 = red
0 = green
11 = blue
10 = exit button
'''
ENTRY_LED_PINS = [13, 6, 5]
EXIT_LED_PINS = [9, 0, 11]
ENTRY_BUTTON_PIN = 19
EXIT_BUTTON_PIN = 10
SERVO_ENTRY_PIN = 23  # physical pin 16
SERVO_EXIT_PIN = 24 # physical pin 

# Setup button pins
GPIO.setup(ENTRY_BUTTON_PIN, GPIO.IN)
GPIO.setup(EXIT_BUTTON_PIN, GPIO.IN)

# Setup LED pins
for pin in ENTRY_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
for pin in EXIT_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)

# Setup servo
GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
GPIO.setup(SERVO_EXIT_PIN, GPIO.OUT)

# Set gate, 0 is entrance, 1 is exit, True is closed, False is open
def set_gate(gate_id: int, close: bool):
    FREQ = 50  # 50 Hz for servo
    PIN = 0
    if gate_id == 0:
        PIN = SERVO_ENTRY_PIN
    elif gate_id == 1:
        PIN = SERVO_EXIT_PIN
    else:
        return

    if close:
        # Close
        angle = 0

    else:
        # Open
        if gate_id == 0:
            angle = 70
        else:
            angle = 90

    # Safety clamp
    angle = max(0, min(180, angle))

    # Convert angle to duty cycle (approximate for SG90)
    duty = 2.5 + (angle / 18.0)  # maps 0–180° to ~2.5–12.5% duty

    # GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)

    pwm = GPIO.PWM(PIN, FREQ)
    pwm.start(0)

    try:
        pwm.ChangeDutyCycle(duty)
        sleep(0.7)  # wait for the servo to reach position
    finally:
        pwm.stop()
        # GPIO.cleanup()

    print(f"Moved servo to {angle}° (duty cycle {duty:.2f}%)")

# --- NEW: GPIO + Sense HAT --------------------------------------------------
# Open Gates
# set_gate(0, False) # Open gate 0 (Entry gate)
# set_gate(1, False) # Open gate 1 (Exit gate)
# sleep(1)
#
# Close Gates
# set_gate(0, True)  # Close gate 0 (Entry gate)
# set_gate(1, True)  # Close gate 1 (Exit gate)
#
# Set LED 0=red, 1=blue, 2=green, GPIO.HIGH is on, GPIO.LOW is off
# GPIO.output(ENTRY_LED_PINS[0], GPIO.HIGH)
#
# Check button for entry
# if GPIO.input(ENTRY_BUTTON_PIN) == GPIO.HIGH:
#
# Check button for exit
# if GPIO.input(EXIT_BUTTON_PIN) == GPIO.HIGH:


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
REQUIRE_ONE_VALID_SAMPLE_FOR_STREAK = False

# Tesseract configuration:
TESS_CFG = "--oem 1 --psm 7 -c " \
           "tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
MIN_SAMPLE_CONF = 40.0   # Tesseract confidence threshold
MIN_FINAL_CONF  = 50.0   # Minimal average confidence to be sent
MIN_FINAL_LEN   = (6, 6) # Allowed length range for acceptable string
MIN_FINAL_SAMPLES = 2    # Need >=2 valid readings

# Image processing
AREA_MIN  = 2000
AREA_MAX  = 5000
AREA_STEP = 1000
AREA_ABS_MIN = 200
area_min  = AREA_MIN
area_max  = AREA_MAX

# Camera Constants
CAMERA_RESOLUTION = (1280, 720)  # or (1920, 1080), (640, 480)

# Anti-Spam controls:
READ_PERIOD = 0.5              # seconds, 2 Hz OCR/evaluation
SIMILAR_TIMEOUT = 10.0         # seconds to block similar/partial plates

# Multi-OCR aggregation controls:
AGGR_MAX_SAMPLES = 5           # collect up to N OCR samples
AGGR_WINDOW = 1.1              # or until this many seconds pass

# Allow sending when the same normalized plate repeats in a streak
LOW_CONF_STREAK_ENABLED = True
LOW_CONF_STREAK_N = 3  # send even if confidence is low

# HW Constants
GPIO_STABLE_MS = 300           # 300 mSec jitter prevention

# Global Anti-spam state variables:
last_sent_plate = ""           # last plate sent to server
next_read_ts = 0.0             # throttle OCR (monotonic time)
BLOCK_TIMEOUT = 10.0
last_sent_plate = ""
last_sent_time = 0.0

# Scan mode controlled by keys/buttons
scan_mode = 'idle'

# OCR aggregation "buckets"
aggr_left  = {'samples': [], 'start_ts': 0.0,
              'streak_plate': '', 'streak_count': 0, 'streak_x': 0,
              'valid_seen_set': set()}
aggr_right = {'samples': [], 'start_ts': 0.0,
              'streak_plate': '', 'streak_count': 0, 'streak_x': 0,
              'valid_seen_set': set()}

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


def smart_swap(chars):
    """
    Function: smart_swap
    Purpose: Replace 'O'->'0' and 'I'->'1' only when adjacent to
    digits (disambiguation). Keep all other characters unchanged.
    Methods: Inspect neighbors; conditionally swap; always append 'ch'.
    Creates: 'out' list with all characters preserved.
    """
    out = []
    for i, ch in enumerate(chars):
        if ch in ("O", "I"):
            left_d = (i > 0 and chars[i-1].isdigit())
            right_d = (i + 1 < len(chars) and chars[i+1].isdigit())
            if left_d or right_d:
                ch = "0" if ch == "O" else "1"
        out.append(ch)
    return "".join(out)


def normalize_plate(txt):
    """
    Function: normalize_plate
    Purpose: Clean up OCR text with minimal ambiguity fixes.
    Methods: Upper-case, trim, remove spaces; keep A-Z0-9 only;
    then call smart_swap(...) to disambiguate O/I near digits.
    Creates: returns normalized string (not None).
    """
    s = txt.upper().strip().replace(" ", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(ch for ch in s if ch in allowed)
    return smart_swap(s)


def update_streak(bucket, plate, x):
    """
    Function: update_streak
    Purpose: Count consecutive repeats of the same normalized plate,
    even when confidence is low; remember the latest x and start_ts.
    Methods: Compare with bucket['streak_plate']; bump/reset count;
    set start_ts when the streak begins.
    Creates: updates 'streak_*' fields and 'start_ts' if needed.
    """
    now = time.monotonic()
    if not bucket['start_ts']:
        bucket['start_ts'] = now
    if not plate:
        # Empty read breaks the streak softly
        return
    if plate == bucket['streak_plate']:
        bucket['streak_count'] += 1
    else:
        bucket['streak_plate'] = plate
        bucket['streak_count'] = 1
    bucket['streak_x'] = int(x)
    print(f"[streak] {bucket['streak_plate']} x{bucket['streak_count']} "
            f"at x={bucket['streak_x']}")


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
    Purpose: Add an OCR sample into the bucket if confidence is OK.
    Methods: Set start_ts on the first sample; append (plate, conf, x, ts)
    when plate is non-empty and conf >= MIN_SAMPLE_CONF.
    Creates: tuples in 'samples': (plate, conf, x, ts).
    """
    now = time.monotonic()
    if not bucket['samples'] and not bucket['start_ts']:
        bucket['start_ts'] = now
    if not plate or float(conf) < MIN_SAMPLE_CONF:
        return
    bucket['samples'].append((plate, float(conf), int(x), now))
    bucket['valid_seen_set'].add(plate)

def choose_best(samples):
    """
    Function: choose_best
    Purpose: Choose the plate with the highest average confidence.
    Methods: Aggregate by plate; compute avg_conf; pick max.
    Creates: returns (best_plate, best_x, best_avg_conf).
    """
    if not samples:
        return "", 0, 0.0
    stats = {}
    for plate, conf, x, ts in samples:
        s = stats.setdefault(plate, {'sum': 0.0, 'n': 0, 'x': x})
        s['sum'] += conf
        s['n'] += 1
        s['x'] = x
    best_plate, best_avg, best_x = "", -1.0, 0
    for p, s in stats.items():
        avg = s['sum'] / max(s['n'], 1)
        if avg > best_avg:
            best_plate, best_avg, best_x = p, avg, s['x']
    return best_plate, best_x, best_avg


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
    Purpose: Finalize when enough samples or window passed; prefer the
    normal path (confidence-filtered). If that fails, allow streak-based
    override: send when the same valid plate repeated N times in a row.
    Methods: Count/time window check; choose_best; strict AAA999 filters;
    streak override; clear bucket before exit in all cases.
    Creates: Calls handle_candidate(); clears bucket afterwards.
    """
    # Need either some samples or at least a started streak window
    if not bucket['samples'] and not bucket['start_ts']:
        return

    now = time.monotonic()
    enough_count = (len(bucket['samples']) >= AGGR_MAX_SAMPLES)
    enough_time = ((now - bucket['start_ts']) >= AGGR_WINDOW)
    if not (enough_count or enough_time):
        return

    # ---- Normal path: confidence-filtered best candidate ----
    plate, x, avg_conf = ("", 0, 0.0)
    if bucket['samples']:
        plate, x, avg_conf = choose_best(bucket['samples'])

    # Strict AAA999
    def strict_ok(p, avg):
        if not p:
            return False
        if not (MIN_FINAL_LEN[0] <= len(p) <= MIN_FINAL_LEN[1]):
            return False
        return bool(RE_PLATE.fullmatch(p)) and (avg >= MIN_FINAL_CONF)

    if strict_ok(plate, avg_conf):
        clear_aggr(bucket)
        handle_candidate(plate, x, frame_width)
        return

    # ---- Streak override: same normalized plate repeated N times ----
    if LOW_CONF_STREAK_ENABLED:
        sp = bucket.get('streak_plate', '')
        sc = int(bucket.get('streak_count', 0))
        sx = int(bucket.get('streak_x', 0))
        if sp and RE_PLATE.fullmatch(sp) and (sc >= LOW_CONF_STREAK_N):
            if (not REQUIRE_ONE_VALID_SAMPLE_FOR_STREAK) or \
            (sp in bucket.get('valid_seen_set', set())):
                clear_aggr(bucket)
                handle_candidate(sp, sx, frame_width)
                return

    # ---- Neither path qualified: clear and do nothing ----
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
def handle_candidate(candidate, x, frame_width):
    """
    Function: handle_candidate
    Purpose: Simple anti-spam: do not resend the same plate within
    BLOCK_TIMEOUT seconds; otherwise send immediately.
    Methods: Compare candidate vs last_sent_plate and time delta;
    call send_plate_event(); update globals.
    Creates: updates last_sent_*.
    """
    global last_sent_plate, last_sent_time
    now = time.monotonic()

    if candidate == last_sent_plate and (now - last_sent_time) < BLOCK_TIMEOUT:
        return

    print(f"The plate location is {x}")
    send_plate_event(candidate, x, frame_width)
    last_sent_plate = candidate
    last_sent_time = now

# -------------------- Multi-OCR aggregation --------------------------------
def clear_aggr(bucket):
    """
    Function: clear_aggr
    Purpose: Reset aggregation bucket to empty state.
    Methods: Clear list and zero start_ts; reset streak fields.
    Creates: empties 'samples', sets 'start_ts' to 0; clears streak.
    """
    bucket['samples'].clear()
    bucket['start_ts'] = 0.0
    bucket['streak_plate'] = ''
    bucket['streak_count'] = 0
    bucket['streak_x'] = 0
    bucket['valid_seen_set'].clear()

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
    # sense = SenseHat()
    # sense.clear()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_ENTER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(PIN_EXIT,  GPIO.IN, pull_up_down=GPIO.PUD_UP)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        GPIO.cleanup()
        # sense.clear()
        return

    print("Press 'e' ENTER-only, 'x' EXIT-only, 's' snapshot, 'z' toggle zones, 'q' quit.")
    print("GPIO4=Exit(blue v), GPIO14=Enter(red ^); both -> EXIT prioritized.")

    next_read_ts = 0.0
    prev_state = (None, None)

    # Camera picture settings
    brightness = 58
    contrast = 148
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
            # show_arrows(sense, enter_low, exit_low)
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
                        update_streak(aggr_left, p, px)
                        add_sample(aggr_left, p, conf, px)
                    maybe_finalize(aggr_left, frame.shape[1])

                if scan_mode == 'enter' and best_right:
                    p, conf, px = ocr_bbox(frame, best_right)
                    if p:
                        update_streak(aggr_right, p, px)
                        add_sample(aggr_right, p, conf, px)
                    maybe_finalize(aggr_right, frame.shape[1])

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
        # sense.clear()
        GPIO.cleanup()

if __name__ == "__main__":
    main()
