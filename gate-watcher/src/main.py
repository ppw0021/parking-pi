'''
GateWatcher v0.0.11 (20251019_1330):
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
 - Adding LEDs, Buttons and Servos

GateWatcher v0.0.9 (20251019):
- Use leds.py (LedControl) for LED policy.
- EXIT_ON_RIGHT flag controls which side is Exit bay.
- LED rules:
  idle -> all OFF
  enter/exit -> BLUE ON on mode change; before sending -> BLUE OFF
  reply 210 -> GREEN ON, open/close gate, then GREEN OFF
  reply 211/212 -> RED ON
  reply 213/error -> RED blink 3x @1 Hz
- OCR robustness: correct Otsu flag + adaptive threshold fallback;
  primary PSM=8 (single word) + fallback PSM=7 (single line).

GateWatcher v0.0.10 (20251019):
- Idle mode: no camera/OCR readings; sleep & wait for buttons.
- Use leds.py (LedControl) for LED policy.
- EXIT_ON_RIGHT flag controls which side is Exit bay.
- LED rules:
  idle -> all OFF
  enter/exit -> BLUE ON on mode change; before sending -> BLUE OFF
  reply 210 -> GREEN ON, open/close gate, then GREEN OFF
  reply 211/212 -> RED ON
  reply 213/error -> RED blink 3x @1 Hz
- OCR robustness: correct Otsu flag + adaptive threshold fallback;
  primary PSM=8 (single word) + fallback PSM=7 (single line).
- Streak override: accept conf=0 when same non-empty plate repeats ≥4x.
- Fix common false positive: drop leading I/1 when pattern is I+AAA999.

GateWatcher v0.0.11 (20251019):
- Video feed is shown in all modes (incl. IDLE).
- Always show video feed; in IDLE do display-only (no OCR/server).
- LedControl usage; EXIT_ON_RIGHT side switch.
- LED rules:
  idle -> all OFF
  enter/exit -> BLUE ON on mode change; before sending -> BLUE OFF
  reply 210 -> GREEN ON, open/close gate, then GREEN OFF
  reply 211/212 -> RED ON
  reply 213/error -> RED blink 3x @1 Hz
- OCR robustness: Otsu (BINARY) + adaptive threshold fallback;
  primary PSM=8 (single word) + fallback PSM=7 (single line).
- Streak override: accept conf=0 when same non-empty plate repeats ≥4x.
- Fix FP: drop leading I/1 when text like I + AAA999.
- Area filter uses bbox area (w*h) and is adjustable with [ / ] keys:
  AREA_MIN=11000, AREA_MAX=18000, STEP=500, ABS_MIN=500, ABS_MAX=40000.
- Suppress only empty OCR log lines (raw=''); keep raw!='' even if conf=0.
- Rotate EXIT ROI by 180° before OCR (plates are upside-down).
- Manual spot selection: press 'a' to use closest bbox to mouse pointer.
- Idle mode enforced if no plate discovered in MAX_TIME_TO_IDLE seconds.
'''
import requests
from time import sleep
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import time
from datetime import datetime
import re
import RPi.GPIO as GPIO
import subprocess

# ---- GPIO base setup ---------------------------------------------------------
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

'''
Entry pinout (BCM)
13 = red
6  = green
5  = blue
19 = entry button

Exit pinout (BCM)
9  = red
0  = green
11 = blue
10 = exit button
'''
ENTRY_LED_PINS = [13, 6, 5]
EXIT_LED_PINS  = [9, 0, 11]
ENTRY_BUTTON_PIN = 19
EXIT_BUTTON_PIN  = 10

SERVO_ENTRY_PIN = 23   # Entry gate servo
SERVO_EXIT_PIN  = 24   # Exit gate servo

# Setup button pins
GPIO.setup(ENTRY_BUTTON_PIN, GPIO.IN)
GPIO.setup(EXIT_BUTTON_PIN,  GPIO.IN)

# Setup LED pins
for pin in ENTRY_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
for pin in EXIT_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)

# Setup servo pins
GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
GPIO.setup(SERVO_EXIT_PIN,  GPIO.OUT)

# --- Led controller (from leds.py) -------------------------------------------
from leds import LedControl
led = LedControl(ENTRY_LED_PINS, EXIT_LED_PINS)

# ---- Configuration -----------------------------------------------------------
CAMERA_INDEX = 0
# WEB_PI_IP = "http://10.138.63.88"  # old location
WEB_PI_IP = "http://192.168.1.16"    # current location
URL = f"{WEB_PI_IP}:5000"

ASPECT_MIN = 2.0
ASPECT_MAX = 6.0
MAX_CANDIDATES = 10
PRINT_ALL_OCR = True

# Zone thresholds (fractions of frame width)
EXIT_ZONE_X_LIMIT  = 0.52
ENTER_ZONE_X_LIMIT = 0.58

# Exit bay side flag: when True, Exit is on the right side
EXIT_ON_RIGHT = True

# Strict plate pattern: AAA999
RE_PLATE = re.compile(r'^[A-Z]{3}\d{3}$')

# Auto-IDLE settings
AUTO_IDLE_ENABLED = True
MAX_TIME_TO_IDLE  = 4.0  # seconds without a spot -> go IDLE

# OCR confidences and rules
MIN_SAMPLE_CONF   = 40.0
MIN_FINAL_CONF    = 50.0
MIN_FINAL_LEN     = (6, 6)
MIN_FINAL_SAMPLES = 2

# Tesseract configs: primary word, fallback line
TESS_CFG_PRIMARY  = (
    "--oem 1 --psm 8 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
TESS_CFG_FALLBACK = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

# Image area settings (bbox area filter controlled by [ / ])
AREA_MIN      = 11000
AREA_MAX      = 18000
AREA_STEP     = 500
AREA_ABS_MIN  = 500
AREA_ABS_MAX  = 40000
area_min = AREA_MIN
area_max = AREA_MAX

# Camera
CAMERA_RESOLUTION = (1280, 720)

# Anti-spam controls
READ_PERIOD    = 0.5  # seconds
SIMILAR_TIMEOUT = 10.0
AGGR_MAX_SAMPLES = 5
AGGR_WINDOW      = 1.1

LOW_CONF_STREAK_ENABLED = True
LOW_CONF_STREAK_N       = 4
REQUIRE_ONE_VALID_SAMPLE_FOR_STREAK = False

# Anti-spam global state
last_sent_plate = ""
next_read_ts = 0.0
BLOCK_TIMEOUT = 10.0
last_sent_plate = ""
last_sent_time = 0.0

# Last time when a valid spot was seen in each bay
last_spot_ts = {'enter': 0.0, 'exit': 0.0}

# Scan mode
scan_mode = 'idle'

# OCR aggregation buckets
aggr_left  = {'samples': [], 'start_ts': 0.0,
              'streak_plate': '', 'streak_count': 0, 'streak_x': 0,
              'valid_seen_set': set()}
aggr_right = {'samples': [], 'start_ts': 0.0,
              'streak_plate': '', 'streak_count': 0, 'streak_x': 0,
              'valid_seen_set': set()}

show_zones = True

# Manual selection (mouse)
last_mouse_pos = None

# ---- Mouse callback ----------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    """
    Function: on_mouse
    Purpose: Track mouse pointer position for manual spot selection.
    Methods: cv2.setMouseCallback; store latest (x, y).
    Creates: updates global last_mouse_pos.
    """
    global last_mouse_pos
    if event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN,
                 cv2.EVENT_RBUTTONDOWN):
        last_mouse_pos = (x, y)

# ---- Helpers ----------------------------------------------------------------
def set_gate(gate_id: int, close: bool):
    """
    Function: set_gate
    Purpose: Move a servo to open/close entry(0)/exit(1) gate.
    Methods: GPIO.PWM at 50 Hz; angle->duty mapping; small dwell.
    Creates: local PWM object (started/stopped within try/finally).
    """
    FREQ = 50
    if gate_id == 0:
        pin = SERVO_ENTRY_PIN
    elif gate_id == 1:
        pin = SERVO_EXIT_PIN
    else:
        return

    if close:
        angle = 0
    else:
        angle = 70 if gate_id == 0 else 90

    angle = max(0, min(180, angle))
    duty = 2.5 + (angle / 18.0)

    pwm = GPIO.PWM(pin, FREQ)
    pwm.start(0)
    try:
        pwm.ChangeDutyCycle(duty)
        sleep(0.7)
    finally:
        pwm.stop()
    print(f"Moved servo to {angle}° (duty {duty:.2f}%)")

def read_gpio_state():
    """
    Function: read_gpio_state
    Purpose: Sample button pins; HIGH means active per wiring.
    Methods: GPIO.input() on ENTRY_BUTTON_PIN/EXIT_BUTTON_PIN.
    Creates: two booleans (enter_high, exit_high).
    """
    enter_high = (GPIO.input(ENTRY_BUTTON_PIN) == GPIO.HIGH)
    exit_high  = (GPIO.input(EXIT_BUTTON_PIN)  == GPIO.HIGH)
    return enter_high, exit_high

def set_brightness(value):
    """
    Function: set_brightness
    Purpose: Control UVC camera brightness via v4l2-ctl.
    Methods: Clamp 0..255; subprocess.call().
    Creates: none.
    """
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0",
        "--set-ctrl", f"brightness={value}"
    ])
    print(f"Brightness: {value}")
    return value

def set_contrast(value):
    """
    Function: set_contrast
    Purpose: Control UVC camera contrast via v4l2-ctl.
    Methods: Clamp 0..255; subprocess.call().
    Creates: none.
    """
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0",
        "--set-ctrl", f"contrast={value}"
    ])
    print(f"Contrast: {value}")
    return value

def set_gain(value):
    """
    Function: set_gain
    Purpose: Control UVC camera gain via v4l2-ctl.
    Methods: Clamp 0..255; subprocess.call().
    Creates: none.
    """
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", "--device=/dev/video0",
        "--set-ctrl", f"gain={value}"
    ])
    print(f"Gain: {value}")
    return value

def preprocess_roi(roi_bgr):
    """
    Function: preprocess_roi
    Purpose: Prepare a plate ROI for OCR with robust binarization.
    Methods: Gray -> bilateral -> CLAHE -> Otsu (BINARY).
             If Otsu bad: adaptive thresh (Gaussian, BINARY); close.
    Creates: 'th' bin image.
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    filt = cv2.bilateralFilter(gray, 7, 25, 25)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    _, th = cv2.threshold(
        clahe.apply(filt), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    fg_ratio = float(np.count_nonzero(th)) / float(th.size)
    if fg_ratio < 0.12 or fg_ratio > 0.88:
        th = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2
        )

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    return th

def find_plate_candidates(frame_bgr):
    """
    Function: find_plate_candidates
    Purpose: Detect rectangular regions that may contain a plate.
    Methods: Gray -> white mask -> Canny -> dilate -> contours;
             bbox area (w*h) filter with area_min/area_max; aspect filter;
             return top-N by bbox area.
    Creates: boxes list of (x, y, w, h).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    combined = cv2.bitwise_and(gray, binary_mask)
    edges = cv2.Canny(combined, 80, 200)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, k, iterations=1)
    cnts, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        bbox_area = int(w) * int(h)
        if bbox_area < area_min or bbox_area > area_max:
            continue
        aspect = w / max(h, 1)
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes[:MAX_CANDIDATES]

def ocr_text_and_conf(img_bin):
    """
    Function: ocr_text_and_conf
    Purpose: Run Tesseract OCR and return text with avg confidence.
    Methods: image_to_data with PRIMARY config; if empty -> FALLBACK;
             mean conf over words with conf>0.
    Creates: returns (raw_text, avg_conf).
    """
    def run(cfg):
        data = pytesseract.image_to_data(
            img_bin, config=cfg, output_type=Output.DICT
        )
        confs = []
        for c in data.get('conf', []):
            try:
                v = int(c)
                if v > 0:
                    confs.append(v)
            except ValueError:
                continue
        avg_conf = float(sum(confs)) / max(len(confs), 1) if confs else 0.0
        raw = " ".join([w for w in data.get('text', []) if w.strip()])
        return raw, avg_conf

    raw, avg = run(TESS_CFG_PRIMARY)
    if not raw.strip():
        raw, avg = run(TESS_CFG_FALLBACK)
    return raw, avg

def ocr_plate(roi_bin):
    """
    Function: ocr_plate
    Purpose: (compat) image_to_string on bin image.
    Methods: pytesseract.image_to_string with PRIMARY config only.
    Creates: txt string.
    """
    txt = pytesseract.image_to_string(roi_bin, config=TESS_CFG_PRIMARY)
    return txt.strip()

def smart_swap(chars):
    """
    Function: smart_swap
    Purpose: Replace 'O'->'0' and 'I'->'1' near digits.
    Methods: neighbor digit checks; otherwise keep char unchanged.
    Creates: 'out' list.
    """
    out = []
    for i, ch in enumerate(chars):
        if ch in ("O", "I"):
            left_d  = (i > 0 and chars[i-1].isdigit())
            right_d = (i + 1 < len(chars) and chars[i+1].isdigit())
            if left_d or right_d:
                ch = "0" if ch == "O" else "1"
        out.append(ch)
    return "".join(out)

def normalize_plate(txt):
    """
    Function: normalize_plate
    Purpose: Cleanup OCR text with minimal ambiguity fixes.
    Methods: upper-case, strip spaces, whitelist A-Z0-9, smart_swap.
    Creates: normalized string.
    """
    s = txt.upper().strip().replace(" ", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(ch for ch in s if ch in allowed)
    return smart_swap(s)

def fix_common_false_positives(p: str) -> str:
    """
    Function: fix_common_false_positives
    Purpose: Drop spurious leading I/1 when text looks like I + AAA999.
             or AAA999 + 1
    Methods: regex check; slice off first.
    Creates: corrected plate or original.
    """
    if len(p) == 7 and re.fullmatch(r'^[I1][A-Z]{3}\d{3}$', p):
        return p[1:]
    if len(p) == 7 and p.endswith('1') and re.fullmatch(r'^[A-Z]{3}\d{4}$', p):
        p = p[:-1]
    return p

def update_streak(bucket, plate, x):
    """
    Function: update_streak
    Purpose: Count consecutive repeats of same plate; keep latest x/ts.
    Methods: compare with bucket['streak_plate']; bump/reset count.
    Creates: updates 'streak_*' fields.
    """
    now = time.monotonic()
    if not bucket['start_ts']:
        bucket['start_ts'] = now
    if not plate:
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
    Purpose: Select best (largest bbox area) for left/right zones.
    Methods: split by x threshold; pick max area per side.
    Creates: best_left, best_right tuples or None.
    """
    left_zone  = [b for b in boxes
                  if b[0] < frame_width * EXIT_ZONE_X_LIMIT]
    right_zone = [b for b in boxes
                  if b[0] > frame_width * ENTER_ZONE_X_LIMIT]

    def area(b): return b[2] * b[3]
    left_zone.sort(key=area, reverse=True)
    right_zone.sort(key=area, reverse=True)
    best_left  = left_zone[0]  if left_zone  else None
    best_right = right_zone[0] if right_zone else None
    return best_left, best_right

def add_sample(bucket, plate, conf, x):
    """
    Function: add_sample
    Purpose: Add OCR sample into the bucket if confidence is OK.
    Methods: start ts on first; append (plate, conf, x, ts) for conf>=min.
    Creates: entries in bucket['samples'] and 'valid_seen_set'.
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
    Purpose: Choose plate with highest average confidence.
    Methods: aggregate by plate; compute avg; pick max.
    Creates: returns (best_plate, best_x, best_avg_conf).
    """
    if not samples:
        return "", 0, 0.0
    stats = {}
    for plate, conf, x, ts in samples:
        s = stats.setdefault(plate, {'sum': 0.0, 'n': 0, 'x': x})
        s['sum'] += conf
        s['n']   += 1
        s['x']    = x
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
    Methods: regex match against RE_PLATE.
    Creates: None.
    """
    return bool(RE_PLATE.fullmatch(plate))

def side_by_x(x, frame_width):
    """
    Function: side_by_x
    Purpose: Map x position to 'enter' or 'exit' using EXIT_ON_RIGHT.
    Methods: compare with zone limits; apply flag mapping.
    Creates: 'enter'/'exit' or 'middle'.
    """
    is_left  = x < frame_width * EXIT_ZONE_X_LIMIT
    is_right = x > frame_width * ENTER_ZONE_X_LIMIT
    if EXIT_ON_RIGHT:
        if is_right: return 'exit'
        if is_left:  return 'enter'
    else:
        if is_left:  return 'exit'
        if is_right: return 'enter'
    return 'middle'

def maybe_finalize(bucket, frame_width):
    """
    Function: maybe_finalize
    Purpose: Finalize when enough samples or time window passed.
             Prefer confidence-filtered path; fallback to streak-based
             override when allowed.
    Methods: choose_best; strict AAA999; optional streak override;
             clear bucket before exit.
    Creates: Calls handle_candidate(); clears bucket.
    """
    if not bucket['samples'] and not bucket['start_ts']:
        return
    now = time.monotonic()
    enough_count = (len(bucket['samples']) >= AGGR_MAX_SAMPLES)
    enough_time  = ((now - bucket['start_ts']) >= AGGR_WINDOW)
    if not (enough_count or enough_time):
        return

    plate, x, avg_conf = ("", 0, 0.0)
    if bucket['samples']:
        plate, x, avg_conf = choose_best(bucket['samples'])
        plate = fix_common_false_positives(plate)

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
    if LOW_CONF_STREAK_ENABLED:
        sp = bucket.get('streak_plate', '')
        sc = int(bucket.get('streak_count', 0))
        sx = int(bucket.get('streak_x', 0))
        sp_fixed = fix_common_false_positives(sp)

        if sp_fixed and RE_PLATE.fullmatch(sp_fixed) and \
           (sc >= LOW_CONF_STREAK_N):
            if (not REQUIRE_ONE_VALID_SAMPLE_FOR_STREAK) or \
               (sp_fixed in bucket.get('valid_seen_set', set())):
                clear_aggr(bucket)
                handle_candidate(sp_fixed, sx, frame_width)
                return
    clear_aggr(bucket)

def ocr_bbox(frame, box, side_hint=None):
    """
    Function: ocr_bbox
    Purpose: OCR a single bbox and return normalized text and conf.
    Methods: crop->rotate for exit->resize->preprocess->image_to_data;
             normalize; conditional logging (skip only empty reads).
    Creates: returns (plate, conf, x_left, x_center).
    """
    (x, y, w, h) = box
    roi = frame[y:y + h, x:x + w]

    # Rotate EXIT ROI by 180 degrees (numbers are upside-down)
    if side_hint == 'exit':
        roi = cv2.rotate(roi, cv2.ROTATE_180)

    roi = cv2.resize(
        roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )
    th = preprocess_roi(roi)
    raw, conf = ocr_text_and_conf(th)
    plate = normalize_plate(raw)
    plate = fix_common_false_positives(plate)

    # Conditional logging: print only when raw is non-empty (keep conf=0)
    if PRINT_ALL_OCR and raw.strip():
        fw = frame.shape[1]
        side = 'C'
        if x < fw * EXIT_ZONE_X_LIMIT:
            side = 'L'
        elif x > fw * ENTER_ZONE_X_LIMIT:
            side = 'R'
        print(f"[side {side}] raw={raw!r} norm={plate!r} conf={conf:.1f}")

    x_center = x + w // 2
    return plate, conf, x, x_center

def draw_zones(vis):
    """
    Function: draw_zones
    Purpose: Draw EXIT/ENTER vertical borders and labels.
    Methods: compute pixel x from relative limits; draw lines and text.
    Creates: draws on 'vis' in-place.
    """
    h, w = vis.shape[:2]
    x_exit  = int(w * EXIT_ZONE_X_LIMIT)
    x_enter = int(w * ENTER_ZONE_X_LIMIT)
    cv2.line(vis, (x_exit, 0), (x_exit, h), (0, 255, 255), 2)
    cv2.line(vis, (x_enter, 0), (x_enter, h), (0, 255, 0), 2)
    cv2.putText(
        vis, f"EXIT <={x_exit}px", (10, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, f"ENTER >={x_enter}px", (10, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
    )

def draw_box_with_area(vis, box, color=(0, 255, 0)):
    """
    Function: draw_box_with_area
    Purpose: Draw bbox and overlay two area labels:
             S = estimated contour area inside ROI,
             A = bbox area (w*h).
    Methods: slice ROI, simple binarization, contours, max area.
    Creates: label near top-left of box.
    """
    x, y, w, h = box[:4]
    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
    bbox_area = int(w) * int(h)
    roi = vis[y:y + h, x:x + w]
    if roi.size > 0:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        filt = cv2.bilateralFilter(gray, 7, 25, 25)
        _, bin_inv = cv2.threshold(
            filt, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, k, iterations=1)
        cnts, _ = cv2.findContours(
            clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_area = max((cv2.contourArea(c) for c in cnts), default=0)
    else:
        contour_area = 0

    label = f"S={int(contour_area)} A={bbox_area}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 2
    (tw, th), base = cv2.getTextSize(label, font, scale, thick)
    tx = x + 4
    ty = max(y + th + 4, th + 4)
    cv2.rectangle(vis, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 2),
                  (0, 0, 0), -1)
    cv2.putText(vis, label, (tx, ty), font, scale,
                (0, 255, 255), thick, cv2.LINE_AA)

def send_plate_event(plate, x, frame_width):
    """
    Function: send_plate_event
    Purpose: Call server URL by side (enter/exit) and act on reply.
    Methods: side_by_x(); requests.get; LED policy and gate cycle.
    Creates: prints and hardware actions.
    """
    try:
        side = side_by_x(x, frame_width)
        if side == 'middle':
            return

        op = side  # 'enter' or 'exit'
        uri = f"{URL}/{op}/{plate}"

        response = requests.get(uri, timeout=5)
        code = response.status_code
        print(f"Sent {uri} -> HTTP {code}")

        if code == 210:
            print(f"{op.title()}: success, open gate.")
            led.green_on(op)
            gate_id = 0 if op == 'enter' else 1
            set_gate(gate_id, False)
            time.sleep(3)
            set_gate(gate_id, True)
            led.green_off(op)

        elif code in (211, 212):
            print(f"{op.title()}: negative reply ({code}).")
            led.red_on(op)

        elif code == 213:
            print(f"{op.title()}: error/invalid plate.")
            led.blink_red(op, times=3, freq_hz=1.0)

        else:
            print(f"{op.title()}: unexpected status {code}.")
            led.blink_red(op, times=3, freq_hz=1.0)

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        side = side_by_x(x, frame_width)
        if side in ('enter', 'exit'):
            led.blink_red(side, times=3, freq_hz=1.0)

def handle_candidate(candidate, x, frame_width):
    """
    Function: handle_candidate
    Purpose: Simple anti-spam; BEFORE sending: BLUE OFF for that side.
    Methods: compare with last_sent; led.blue_off(side); send.
    Creates: updates last_sent_* globals.
    """
    global last_sent_plate, last_sent_time
    now = time.monotonic()
    if candidate == last_sent_plate and (now - last_sent_time) < BLOCK_TIMEOUT:
        return

    side = side_by_x(x, frame_width)
    if side in ('enter', 'exit'):
        led.blue_off(side)

    send_plate_event(candidate, x, frame_width)
    last_sent_plate = candidate
    last_sent_time = now

def clear_aggr(bucket):
    """
    Function: clear_aggr
    Purpose: Reset aggregation bucket to empty state.
    Methods: clear list; zero start_ts; reset streak fields/sets.
    Creates: empties samples, start_ts, streak and valid_seen_set.
    """
    bucket['samples'].clear()
    bucket['start_ts'] = 0.0
    bucket['streak_plate'] = ''
    bucket['streak_count'] = 0
    bucket['streak_x'] = 0
    bucket['valid_seen_set'].clear()

def toggle_mode(new_mode):
    """
    Function: toggle_mode
    Purpose: Toggle 'idle'/'enter'/'exit'; clear aggregators; drive LEDs.
    Methods: Switch logic; LED policy:
             idle -> all OFF; enter/exit -> BLUE ON for the side.
    Creates: updates global scan_mode and buckets.
    """
    global scan_mode
    global aggr_left, aggr_right
    if scan_mode == new_mode:
        scan_mode = 'idle'
    else:
        scan_mode = new_mode
    clear_aggr(aggr_left)
    clear_aggr(aggr_right)

    led.all_off()
    now = time.monotonic()
    if scan_mode == 'enter':
        led.blue_on('enter')
        last_spot_ts['enter'] = now  # start enter no-spot timer
    elif scan_mode == 'exit':
        led.blue_on('exit')
        last_spot_ts['exit'] = now  # start exit no-spot timer

    print(f"Scan mode: {scan_mode.upper() if scan_mode!='idle' else 'IDLE'}")
    time.sleep(0.2)

# ---- Manual selection --------------------------------------------------------
def manual_select_and_process(frame, boxes):
    """
    Function: manual_select_and_process
    Purpose: On 'a' key, pick bbox closest to mouse pointer and send it.
    Methods: nearest by center distance; side_by_x decides rotation hint;
             ocr_bbox -> handle_candidate (uses anti-spam).
    Creates: triggers one immediate processing for the chosen box.
    """
    global last_mouse_pos
    if last_mouse_pos is None or not boxes:
        print("Manual select: no mouse or no boxes.")
        return

    mx, my = last_mouse_pos
    def center(b): return (b[0] + b[2] // 2, b[1] + b[3] // 2)
    # Pick nearest by Euclidean distance
    best = min(
        boxes,
        key=lambda b: (center(b)[0] - mx) ** 2 + (center(b)[1] - my) ** 2
    )
    cx = best[0] + best[2] // 2
    side = side_by_x(cx, frame.shape[1])
    p, conf, x_left, x_center = ocr_bbox(frame, best, side_hint=side)
    if p:
        # Use center X for side decision robustness
        handle_candidate(p, x_center, frame.shape[1])
    else:
        print("Manual select: OCR empty for chosen box.")

# ---- Main loop ---------------------------------------------------------------
def main():
    """
    Function: main
    Purpose: Always show video feed; in IDLE do display-only (no OCR,
             no plate detection, no server); in active modes run full
             pipeline with 2 Hz OCR and anti-spam; support manual select.
    Methods: VideoCapture opened at start; per-mode branching.
    Creates: cap, HUD, throttling timestamps.
    """
    global next_read_ts, scan_mode, show_zones
    print("Press 'e' ENTER-only, 'x' EXIT-only, 's' snapshot, "
          "'z' toggle zones, 'a' manual spot, '['/']' area range, 'q' quit.")
    print("GPIO4=Exit(blue v), GPIO14=Enter(red ^); both -> EXIT prioritized.")

    # Open camera immediately so the window appears in IDLE too
    cv2.namedWindow("Gate Watcher")
    cv2.setMouseCallback("Gate Watcher", on_mouse)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        GPIO.cleanup()
        return
    next_read_ts = 0.0
    last_spot_ts['enter'] = time.monotonic()
    last_spot_ts['exit']  = time.monotonic()
    brightness = 125
    contrast   = 170

    global area_min, area_max

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            # Read buttons -> set mode
            enter_high, exit_high = read_gpio_state()
            if exit_high:
                toggle_mode('exit')
            elif enter_high:
                toggle_mode('enter')

            # ----- IDLE: display only, skip detection/OCR/server ----------
            if scan_mode == 'idle':
                vis = frame.copy()
                hud = "MODE: IDLE"
                color = (200, 200, 200)
                cv2.putText(
                    vis, hud, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
                )
                cv2.imshow("Gate Watcher", vis)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Exiting")
                    break
                elif key == ord('s'):
                    name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(name, vis)
                    print(f"Frame saved into {name}")
                elif key == ord('e'):
                    toggle_mode('enter')
                elif key == ord('x'):
                    toggle_mode('exit')
                elif key == ord('z'):
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
                    area_max = max(area_min, area_max - AREA_STEP)
                    print(f"Area range: {area_min}–{area_max}")
                elif key == ord(']'):
                    area_min = min(AREA_ABS_MAX, area_min + AREA_STEP)
                    area_max = min(AREA_ABS_MAX, max(area_min, area_max + AREA_STEP))
                    print(f"Area range: {area_min}–{area_max}")
                elif key == ord('a'):
                    # Manual works also in idle (quick single-shot)
                    boxes = find_plate_candidates(frame)
                    manual_select_and_process(frame, boxes)
                continue  # skip heavy processing

            # ----- Active modes: full pipeline ----------------------------
            boxes = find_plate_candidates(frame)
            best_left, best_right = pick_best_by_side(boxes, frame.shape[1])

            # Map boxes to current mode considering EXIT_ON_RIGHT
            best_exit  = best_right if EXIT_ON_RIGHT else best_left
            best_enter = best_left  if EXIT_ON_RIGHT else best_right

            now = time.monotonic()
            # --- Auto-IDLE: if no spots in active bay for MAX_TIME_TO_IDLE, go IDLE
            if AUTO_IDLE_ENABLED:
                if scan_mode == 'enter':
                    if best_enter:
                        # spot present -> refresh last seen time
                        last_spot_ts['enter'] = now
                    elif (now - last_spot_ts['enter']) >= MAX_TIME_TO_IDLE:
                        print(f"Auto-IDLE: no ENTER spots for {MAX_TIME_TO_IDLE:.1f}s")
                        toggle_mode('enter')  # enter->idle per toggle semantics
                elif scan_mode == 'exit':
                    if best_exit:
                        last_spot_ts['exit'] = now
                    elif (now - last_spot_ts['exit']) >= MAX_TIME_TO_IDLE:
                        print(f"Auto-IDLE: no EXIT spots for {MAX_TIME_TO_IDLE:.1f}s")
                        toggle_mode('exit')   # exit->idle per toggle semantics
            if now >= next_read_ts:
                next_read_ts = now + READ_PERIOD

                if scan_mode == 'exit' and best_exit:
                    p, conf, x_left, x_center = ocr_bbox(
                        frame, best_exit, side_hint='exit'
                    )
                    if p:
                        update_streak(aggr_left, p, x_center)
                        add_sample(aggr_left, p, conf, x_center)
                        maybe_finalize(aggr_left, frame.shape[1])

                if scan_mode == 'enter' and best_enter:
                    p, conf, x_left, x_center = ocr_bbox(
                        frame, best_enter, side_hint='enter'
                    )
                    if p:
                        update_streak(aggr_right, p, x_center)
                        add_sample(aggr_right, p, conf, x_center)
                        maybe_finalize(aggr_right, frame.shape[1])

            # Manual selection key in active modes
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                manual_select_and_process(frame, boxes)

            # Visualization
            vis = frame.copy()
            if show_zones:
                draw_zones(vis)
            for (x, y, w, h) in boxes:
                draw_box_with_area(vis, (x, y, w, h))

            hud = "MODE: "
            if scan_mode == 'enter':
                hud += "ENTER (Right)" if not EXIT_ON_RIGHT else "ENTER (Left)"
                color = (0, 255, 0)
            else:  # 'exit'
                hud += "EXIT (Left)" if not EXIT_ON_RIGHT else "EXIT (Right)"
                color = (0, 255, 255)

            cv2.putText(
                vis, hud, (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA
            )
            cv2.imshow("Gate Watcher", vis)

            # Other keys
            if key == ord('q'):
                print("Exiting")
                break
            elif key == ord('s'):
                name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, vis)
                print(f"Frame saved into {name}")
            elif key == ord('e'):
                toggle_mode('enter')
            elif key == ord('x'):
                toggle_mode('exit')
            elif key == ord('z'):
                show_zones = not show_zones
                print(f"Zone overlay: {'ON' if show_zones else 'OFF'}")
            elif key == ord('['):
                area_min = max(AREA_ABS_MIN, area_min - AREA_STEP)
                area_max = max(area_min, area_max - AREA_STEP)
                print(f"Area range: {area_min}–{area_max}")
            elif key == ord(']'):
                area_min = min(AREA_ABS_MAX, area_min + AREA_STEP)
                area_max = min(AREA_ABS_MAX, max(area_min, area_max + AREA_STEP))
                print(f"Area range: {area_min}–{area_max}")
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

    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    main()