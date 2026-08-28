'''
GateWatcher v0.0.13 (20260828_1745):
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

GateWatcher v0.0.12 (20251019):
 - reply 212 from the server is an error and will be handled as such:
  reply 213/212 errors -> RED blink 3x @1 Hz

GateWatcher v0.0.13 (20260828):
- Manual USB camera calibration added.
- Disable camera auto exposure, auto white balance and
  dynamic frame rate control.
- Fine Tune window now includes camera parameter sliders:
  brightness, contrast, exposure, gain,
  white balance, saturation, sharpness and backlight compensation.
- Camera settings are applied immediately via v4l2-ctl.
- ROI is cropped before OCR to remove plate borders and
  surrounding background (CROP_PERCENTAGE).
- OCR debug windows added:
  "OCR ROI" and "OCR TH".
- print_fine_tune_settings() now outputs camera and OCR settings.
- Detection thresholds updated:
  AREA_MIN=15000, AREA_MAX=35000,
  ASPECT_MIN=2.5, ASPECT_MAX=5.0.
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
# ------------- GPIO base setup ---------------------------------------------
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
SERVO_ENTRY_PIN  = 23  # Entry gate servo
SERVO_EXIT_PIN   = 24  # Exit gate servo
# Gate deay
GATE_DELAY = 5
# Exit bay side flag: when True, Exit is on the right side
EXIT_ON_RIGHT = False
# Hardware controls
HW_CONTROL_ENABLED = False
# Setup button pins
GPIO.setup(
    ENTRY_BUTTON_PIN,
    GPIO.IN,
    pull_up_down=GPIO.PUD_UP,
)
GPIO.setup(
    EXIT_BUTTON_PIN,
    GPIO.IN,
    pull_up_down=GPIO.PUD_UP,
)
# Setup LED pins
for pin in ENTRY_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
for pin in EXIT_LED_PINS:
    GPIO.setup(pin, GPIO.OUT)
# Setup servo pins
GPIO.setup(SERVO_ENTRY_PIN, GPIO.OUT)
GPIO.setup(SERVO_EXIT_PIN,  GPIO.OUT)
# --- Led controller (from leds.py) -----------------------------------------
from leds import LedControl
led = LedControl(ENTRY_LED_PINS, EXIT_LED_PINS)
# ---------------- Configuration --------------------------------------------
CAMERA_INDEX = 0
CAMERA_DEVICE = f"/dev/video{CAMERA_INDEX}"
# WEB_PI_IP = "http://10.138.63.88" # old location
# WEB_PI_IP = "http://192.168.1.16"    # current location
# WEB_PI_IP =  "http://10.130.1.228"
WEB_PI_IP =  "http://10.0.0.2"
URL = f"{WEB_PI_IP}:5000"
CROP_PERCENTAGE = 0.10
ASPECT_MIN = 2.5
ASPECT_MAX = 5.0
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
AUTO_IDLE_ENABLED   = False
MAX_TIME_TO_IDLE    = 10.0  # seconds without a spot -> go IDLE
# OCR confidences and rules
MIN_SAMPLE_CONF = 40.0
MIN_FINAL_CONF  = 50.0
MIN_FINAL_LEN   = (6, 6)
MIN_FINAL_SAMPLES = 1
# NEW: softer threshold for 6-char alias keys
MIN_SAMPLE_CONF_ALIAS = 25.0
# Tesseract configs: primary word, fallback line
TESS_CFG_PRIMARY = (
    "--oem 1 --psm 8 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
TESS_CFG_FALLBACK = (
    "--oem 1 --psm 7 "
    "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)
# Image area settings (bbox area filter controlled by [ / ])
AREA_MIN     = 15000
AREA_MAX     = 35000
AREA_STEP    = 500
AREA_ABS_MIN = 500
AREA_ABS_MAX = 40000
area_min = AREA_MIN
area_max = AREA_MAX
CONTRAST = 245
BRIGHTNESS = 85

# Manual camera controls
CAMERA_AUTO_EXPOSURE = 1          # Manual mode
CAMERA_EXPOSURE = 30
CAMERA_GAIN = 140
CAMERA_WB_AUTO = 0
CAMERA_WB_TEMP = 4140
CAMERA_SATURATION = 128
CAMERA_SHARPNESS = 128
CAMERA_BACKLIGHT = 1

# Fine-tune image processing settings
DARK_THRESHOLD = 30
LIGHT_THRESHOLD = 220
CANNY_LOW = 64
CANNY_HIGH = 200
DILATION_ITERATIONS = 2
FINE_TUNE_PRINT_PERIOD = 0.5
FINE_TUNE_WINDOW = "Fine Tune Masks"
MAIN_WINDOW = "Gate Watcher"

# Camera
CAMERA_RESOLUTION = (1280, 720)
# Anti-spam controls
READ_PERIOD     = 0.5  # seconds
SIMILAR_TIMEOUT = 10.0
AGGR_MAX_SAMPLES = 5
AGGR_WINDOW      = 1.1
LOW_CONF_STREAK_ENABLED = True
LOW_CONF_STREAK_N       = 4
REQUIRE_ONE_VALID_SAMPLE_FOR_STREAK = False
# Anti-spam global state
last_sent_plate = ""
next_read_ts    = 0.0
BLOCK_TIMEOUT   = 10.0
last_sent_plate = ""
last_sent_time  = 0.0
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
# Fine-tune state
fine_tune_active = False
fine_tune_dragging = False
fine_tune_start = None
fine_tune_rect = None
fine_tune_last_print = 0.0
fine_tune_input_mode = None
fine_tune_input_text = ""
fine_tune_trackbar_update = False
main_keyboard_active = True

# ---- Mouse callback --------------------------------------------------------
def on_mouse(event, x, y, flags, param):
    """
    Function: on_mouse
    Purpose: Track the pointer and select a calibration rectangle.
    Methods: Store mouse coordinates and process drag events while the
             fine-tune mode is active.
    Creates: Updates last_mouse_pos and fine-tune rectangle state.
    """
    global last_mouse_pos, main_keyboard_active
    global fine_tune_dragging, fine_tune_start, fine_tune_rect

    main_keyboard_active = True

    if event in (
        cv2.EVENT_MOUSEMOVE,
        cv2.EVENT_LBUTTONDOWN,
        cv2.EVENT_LBUTTONUP,
        cv2.EVENT_RBUTTONDOWN,
    ):
        last_mouse_pos = (x, y)

    if not fine_tune_active:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        fine_tune_dragging = True
        fine_tune_start = (x, y)
        fine_tune_rect = (x, y, 0, 0)
    elif event == cv2.EVENT_MOUSEMOVE and fine_tune_dragging:
        x0, y0 = fine_tune_start
        fine_tune_rect = (
            min(x0, x),
            min(y0, y),
            abs(x - x0),
            abs(y - y0),
        )
    elif event == cv2.EVENT_LBUTTONUP and fine_tune_dragging:
        fine_tune_dragging = False
        x0, y0 = fine_tune_start
        fine_tune_rect = (
            min(x0, x),
            min(y0, y),
            abs(x - x0),
            abs(y - y0),
        )
        apply_fine_tune_selection(param, fine_tune_rect)

# ---- Helpers ---------------------------------------------------------------
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
    Purpose: Read active-low entry and exit hardware buttons.
    Methods: Read GPIO inputs configured with internal pull-up resistors.
    Creates: Two booleans indicating whether each button is pressed.
    """
    enter_pressed = (
        GPIO.input(ENTRY_BUTTON_PIN) == GPIO.LOW
    )
    exit_pressed = (
        GPIO.input(EXIT_BUTTON_PIN) == GPIO.LOW
    )

    return enter_pressed, exit_pressed

def set_brightness(value):
    """
    Function: set_brightness
    Purpose: Control UVC camera brightness via v4l2-ctl.
    Methods: Clamp 0..255; subprocess.call().
    Creates: none.
    """
    value = max(0, min(255, value))
    subprocess.call([
        "v4l2-ctl", f"--device={CAMERA_DEVICE}",
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
        "v4l2-ctl", f"--device={CAMERA_DEVICE}",
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
        "v4l2-ctl", f"--device={CAMERA_DEVICE}",
        "--set-ctrl", f"gain={value}"
    ])
    print(f"Gain: {value}")
    return value

def set_camera_ctrl(name, value):
    """
    Function: set_camera_ctrl
    Purpose: Set any UVC control through v4l2-ctl.
    Methods: subprocess.call().
    Creates: none.
    """
    subprocess.call([
        "v4l2-ctl",
        f"--device={CAMERA_DEVICE}",
        "--set-ctrl",
        f"{name}={value}"
    ])
    print(f"{name}: {value}")


def disable_camera_auto_controls():
    """
    Function: disable_camera_auto_controls
    Purpose: Disable camera auto tuning and apply manual values.
    """

    set_camera_ctrl("white_balance_automatic", 0)
    set_camera_ctrl("auto_exposure", 1)
    set_camera_ctrl("exposure_dynamic_framerate", 0)

    set_camera_ctrl(
        "white_balance_temperature",
        CAMERA_WB_TEMP,
    )

    set_camera_ctrl(
        "exposure_time_absolute",
        CAMERA_EXPOSURE,
    )

    set_camera_ctrl("gain", CAMERA_GAIN)
    set_camera_ctrl("saturation", CAMERA_SATURATION)
    set_camera_ctrl("sharpness", CAMERA_SHARPNESS)
    set_camera_ctrl(
        "backlight_compensation",
        CAMERA_BACKLIGHT,
    )

    set_brightness(BRIGHTNESS)
    set_contrast(CONTRAST)

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
def make_detection_masks(frame_bgr):
    """
    Function: make_detection_masks
    Purpose: Build every mask used by the plate candidate detector.
    Methods: Convert to grayscale, threshold dark and light pixels,
             combine both masks, detect edges, and dilate the result.
    Creates: Dark, light, edge, and combined binary masks.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    dark_mask = cv2.inRange(gray, 0, DARK_THRESHOLD)
    light_mask = cv2.inRange(gray, LIGHT_THRESHOLD, 255)
    contrast_mask = cv2.bitwise_or(dark_mask, light_mask)
    edges = cv2.Canny(contrast_mask, CANNY_LOW, CANNY_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined = cv2.dilate(
        edges,
        kernel,
        iterations=max(0, DILATION_ITERATIONS),
    )
    return dark_mask, light_mask, edges, combined


def find_plate_candidates(frame_bgr):
    """
    Function: find_plate_candidates
    Purpose: Detect rectangular regions that may contain a plate.
    Methods: Build adjustable masks, find external contours, apply area
             and aspect filters, then retain the largest candidates.
    Creates: A list of candidate boxes in (x, y, w, h) format.
    """
    _, _, _, combined = make_detection_masks(frame_bgr)
    contours, _ = cv2.findContours(
        combined,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = int(w) * int(h)
        if bbox_area < area_min or bbox_area > area_max:
            continue
        aspect = w / max(h, 1)
        if ASPECT_MIN <= aspect <= ASPECT_MAX:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda box: box[2] * box[3], reverse=True)
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
# ------------------- NEW PIPELINE UTILITIES ---------------------------------
def smart_swap(chars):
    """
    Function: smart_swap
    Purpose: Coerce a 6-char string toward strict 'AAA999' by
             converting left 3 to letters and right 3 to digits.
    Methods: For i<3 convert any digits via DIGIT_TO_LET; for i>=3
             convert any letters via LET_TO_DIG; otherwise keep.
    Creates: Local maps (DIGIT_TO_LET, LET_TO_DIG) and list 'out'.
    """
    s = chars
    if len(s) != 6:
        return s
    DIGIT_TO_LET = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'E', '4': 'A',
        '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G',
    }
    LET_TO_DIG = {
        'O': '0', 'I': '1', 'Z': '2', 'E': '3', 'A': '4',
        'S': '5', 'G': '6', 'T': '7', 'B': '8', 'Q': '0',
        'L': '1',
    }
    out = []
    for i, ch in enumerate(s):
        if i < 3:
            if ch.isdigit():
                out.append(DIGIT_TO_LET.get(ch, ch))
            else:
                out.append(ch)
        else:
            if ch.isalpha():
                out.append(LET_TO_DIG.get(ch, ch))
            else:
                out.append(ch)
    return "".join(out)
def normalize_unicode_lookalikes(text):
    """
    Function: normalize_unicode_lookalikes
    Purpose: Map common Unicode look-alikes for slashed zero and
             similar glyphs to ASCII '0' or 'O' before filtering.
    Methods: Character-by-character replacement using a dict map.
    Creates: Returns a new string with substitutions applied.
    """
    UNI_MAP = {
        '\u00D8': '0',  # Ø LATIN CAPITAL O WITH STROKE
        '\u00F8': '0',  # ø LATIN SMALL O WITH STROKE
        '\u2205': '0',  # ∅ EMPTY SET
        '\u2300': '0',  # ⌀ DIAMETER SIGN
        '\u0398': '0',  # Θ GREEK CAPITAL THETA
        '\u03B8': '0',  # θ GREEK SMALL THETA
        '\u03A6': '0',  # Φ GREEK CAPITAL PHI
        '\u03C6': '0',  # φ GREEK SMALL PHI
        '\u0660': '0',  # ٠ Arabic-Indic digit zero
        '\u06F0': '0',  # ۰ Eastern-Arabic digit zero
    }
    out = []
    for ch in text:
        out.append(UNI_MAP.get(ch, ch))
    return "".join(out)
def normalize_plate(txt):
    """
    Function: normalize_plate
    Purpose: Normalize raw OCR text into a strict 6-char candidate,
             stripping side artifacts and coercing to 'AAA999'.
    Methods:
      0) Unicode look-alike map for slashed zero -> '0' before filter.
      1) Upper-case, strip spaces; whitelist A-Z0-9.
      2) If len is 7 or 8, remove side artifacts 'I','1','J' on edges.
         If still 7 and edges not artifacts -> reject ('').
      3) Require final len==6 -> otherwise ''.
      4) smart_swap to coerce halves (letters/digits).
      5) Validate against RE_PLATE -> return '' if not match.
    Creates: normalized 6-char plate or ''.
    """
    s = normalize_unicode_lookalikes(txt)
    s = s.upper().strip().replace(" ", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(ch for ch in s if ch in allowed)
    n = len(s)
    if n in (7, 8):
        left = s[0]
        right = s[-1]
        if left in ("I", "1", "J"):
            s = s[1:]
            n -= 1
        if n in (7, 8) and right in ("I", "1", "J"):
            s = s[:-1]
            n -= 1
        if n == 7:
            if s[0] not in ("I", "1", "J") and s[-1] not in ("I", "1", "J"):
                return ""
    if len(s) != 6:
        return ""
    s = smart_swap(s)
    if not RE_PLATE.fullmatch(s):
        return ""
    return s
def alias_key_for_streak(txt):
    """
    Function: alias_key_for_streak
    Purpose: Build a 6-char alias key collapsing look-alikes by
             halves (letters left, digits right), ignoring RE_PLATE.
    Methods: Upper/strip/whitelist; strip side artifacts I/1/J at
             edges for len 7-8; require len==6; apply smart_swap.
    Creates: Returns 6-char alias or '' if cannot coerce.
    """
    s = txt.upper().strip().replace(" ", "")
    allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    s = "".join(ch for ch in s if ch in allowed)
    n = len(s)
    if n in (7, 8):
        if s and s[0] in ("I", "1", "J"):
            s = s[1:]; n -= 1
        if s and s[-1] in ("I", "1", "J"):
            s = s[:-1]; n -= 1
    if len(s) != 6:
        return ""
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
    if len(p) == 7 and re.fullmatch(r'^[J1][A-Z]{3}\d{3}$', p):
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
    left_zone.sort(key=area,  reverse=True)
    right_zone.sort(key=area, reverse=True)
    best_left  = left_zone[0]  if left_zone  else None
    best_right = right_zone[0] if right_zone else None
    return best_left, best_right
def add_sample(bucket, plate, conf, x):
    """
    Function: add_sample
    Purpose: Add OCR sample into the bucket if confidence is OK.
    Methods: start ts on first; append (plate, conf, x, ts) for
             conf>=min (use softer threshold when alias present).
    Creates: entries in bucket['samples'] and 'valid_seen_set'.
    """
    now = time.monotonic()
    if not bucket['samples'] and not bucket['start_ts']:
        bucket['start_ts'] = now
    if not plate:
        return
    th = MIN_SAMPLE_CONF_ALIAS if len(plate) == 6 else MIN_SAMPLE_CONF
    if float(conf) < th:
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
        plate = normalize_plate(plate)
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
        sp_fixed = normalize_plate(sp_fixed)
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
    build alias (position-aware) and strict variants; print both;
    forward alias downstream.
    Creates: returns (plate_alias, conf, x_left, x_center).
    """
    x, y, w, h = box

    pad_x = max(4, int(w * 0.08))
    pad_y = max(4, int(h * 0.12))

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)

    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]

    crop_x = int(w * CROP_PERCENTAGE)
    crop_y = int(h * CROP_PERCENTAGE*1.5)

    roi = roi[
        crop_y:h - crop_y,
        crop_x:w - crop_x
    ]
    if side_hint == 'exit':
        roi = cv2.rotate(roi, cv2.ROTATE_180)
    roi = cv2.resize(
        roi, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC
    )
    th = preprocess_roi(roi)
    cv2.imshow("OCR ROI", roi) # DBG
    cv2.imshow("OCR TH", th)   # DBG
    raw, conf = ocr_text_and_conf(th)
    plate_strict = normalize_plate(raw)
    plate_alias  = alias_key_for_streak(raw)
    if PRINT_ALL_OCR and raw.strip():
        fw = frame.shape[1]
        side = 'C'
        if x < fw * EXIT_ZONE_X_LIMIT:
            side = 'L'
        elif x > fw * ENTER_ZONE_X_LIMIT:
            side = 'R'
        print(f"[side {side}] raw={raw!r} alias={plate_alias!r} "
              f"strict={plate_strict!r} conf={conf:.1f}")
    x_center = x + w // 2
    return plate_alias, conf, x, x_center
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
    cv2.line(vis, (x_exit, 0),  (x_exit, h),  (0, 255, 255), 2)
    cv2.line(vis, (x_enter, 0), (x_enter, h), (0, 255, 0),   2)
    cv2.putText(
        vis, f"EXIT <={x_exit}px", (10, h - 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
    )
    cv2.putText(
        vis, f"ENTER >={x_enter}px", (10, h - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   2, cv2.LINE_AA
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
            time.sleep(GATE_DELAY)
            set_gate(gate_id, True)
            led.green_off(op)
        elif code == 211:
            print(f"{op.title()}: negative reply ({code}).")
            led.red_on(op)
        elif code in (213, 212):
            print(f"{op.title()}: error/invalid plate.")
            led.blink_red(op, times=3, freq_hz=1.0)
            if scan_mode in ('enter', 'exit'):
                led.blink_red(side, times=3, freq_hz=1.0)
                led.blue_on(scan_mode)
        else:
            print(f"{op.title()}: unexpected status {code}.")
            led.blink_red(op, times=3, freq_hz=1.0)
            if scan_mode in ('enter', 'exit'):
                led.blink_red(side, times=3, freq_hz=1.0)
                led.blue_on(scan_mode)
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        side = side_by_x(x, frame_width)
        if side in ('enter', 'exit'):
            led.blink_red(side, times=3, freq_hz=1.0)
        if scan_mode in ('enter', 'exit'):
            led.blue_on(scan_mode)
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
    last_sent_time  = now
def clear_aggr(bucket):
    """
    Function: clear_aggr
    Purpose: Reset aggregation bucket to empty state.
    Methods: clear list; zero start_ts; reset streak fields/sets.
    Creates: empties samples, start_ts, streak and valid_seen_set.
    """
    bucket['samples'].clear()
    bucket['start_ts']   = 0.0
    bucket['streak_plate'] = ''
    bucket['streak_count'] = 0
    bucket['streak_x']     = 0
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
        last_spot_ts['exit'] = now   # start exit no-spot timer
    print(f"Scan mode: {scan_mode.upper() if scan_mode!='idle' else 'IDLE'}")
    time.sleep(0.2)
# ---- Manual selection ------------------------------------------------------
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
    best = min(
        boxes,
        key=lambda b: (center(b)[0] - mx) ** 2 + (center(b)[1] - my) ** 2
    )
    cx = best[0] + best[2] // 2
    side = side_by_x(cx, frame.shape[1])
    p, conf, x_left, x_center = ocr_bbox(frame, best, side_hint=side)
    if p:
        handle_candidate(p, x_center, frame.shape[1])
    else:
        print("Manual select: OCR empty for chosen box.")
def on_mask_mouse(event, x, y, flags, param):
    """
    Function: on_mask_mouse
    Purpose: Disable hotkeys while the mask window is being used.
    Methods: Mark the main window inactive on any mask-window mouse event.
    Creates: Updated main_keyboard_active state.
    """
    global main_keyboard_active

    main_keyboard_active = False


def fine_tune_trackbar_changed(_value):
    """
    Function: fine_tune_trackbar_changed
    Purpose: Copy trackbar positions into processing settings.
    Methods: Read all controls and enforce valid minimum/maximum pairs.
    Creates: Updated global image-processing constants.
    """
    global BRIGHTNESS
    global CONTRAST, CAMERA_EXPOSURE, CAMERA_GAIN
    global CAMERA_WB_TEMP, CAMERA_SATURATION
    global CAMERA_SHARPNESS, CAMERA_BACKLIGHT
    global DARK_THRESHOLD, LIGHT_THRESHOLD
    global CANNY_LOW, CANNY_HIGH, DILATION_ITERATIONS
    global ASPECT_MIN, ASPECT_MAX, area_min, area_max
    global fine_tune_trackbar_update, main_keyboard_active

    main_keyboard_active = False
    if fine_tune_trackbar_update:
        return

    BRIGHTNESS = cv2.getTrackbarPos(
        "Brightness",
        FINE_TUNE_WINDOW,
    )

    CONTRAST = cv2.getTrackbarPos(
        "Contrast",
        FINE_TUNE_WINDOW,
    )

    CAMERA_EXPOSURE = cv2.getTrackbarPos(
        "Exposure",
        FINE_TUNE_WINDOW,
    )

    CAMERA_GAIN = cv2.getTrackbarPos(
        "Gain",
        FINE_TUNE_WINDOW,
    )

    CAMERA_WB_TEMP = cv2.getTrackbarPos(
        "WB Temp",
        FINE_TUNE_WINDOW,
    )

    CAMERA_SATURATION = cv2.getTrackbarPos(
        "Saturation",
        FINE_TUNE_WINDOW,
    )

    CAMERA_SHARPNESS = cv2.getTrackbarPos(
        "Sharpness",
        FINE_TUNE_WINDOW,
    )

    CAMERA_BACKLIGHT = cv2.getTrackbarPos(
        "Backlight",
        FINE_TUNE_WINDOW,
    )

    set_brightness(BRIGHTNESS)
    set_contrast(CONTRAST)

    set_camera_ctrl(
        "exposure_time_absolute",
        CAMERA_EXPOSURE,
    )

    set_camera_ctrl(
        "white_balance_temperature",
        CAMERA_WB_TEMP,
    )

    set_camera_ctrl(
        "gain",
        CAMERA_GAIN,
    )

    set_camera_ctrl(
        "saturation",
        CAMERA_SATURATION,
    )

    set_camera_ctrl(
        "sharpness",
        CAMERA_SHARPNESS,
    )

    set_camera_ctrl(
       "backlight_compensation",
        CAMERA_BACKLIGHT,
    )

    DARK_THRESHOLD = cv2.getTrackbarPos("Dark", FINE_TUNE_WINDOW)
    LIGHT_THRESHOLD = cv2.getTrackbarPos("Light", FINE_TUNE_WINDOW)
    CANNY_LOW = cv2.getTrackbarPos("Canny low", FINE_TUNE_WINDOW)
    CANNY_HIGH = cv2.getTrackbarPos("Canny high", FINE_TUNE_WINDOW)
    DILATION_ITERATIONS = cv2.getTrackbarPos(
        "Dilation",
        FINE_TUNE_WINDOW,
    )
    area_min = max(
        AREA_ABS_MIN,
        cv2.getTrackbarPos("Area min", FINE_TUNE_WINDOW),
    )
    area_max = max(
        area_min,
        cv2.getTrackbarPos("Area max", FINE_TUNE_WINDOW),
    )
    ASPECT_MIN = max(
        0.1,
        cv2.getTrackbarPos("Aspect min x100", FINE_TUNE_WINDOW) / 100.0,
    )
    ASPECT_MAX = max(
        ASPECT_MIN,
        cv2.getTrackbarPos("Aspect max x100", FINE_TUNE_WINDOW) / 100.0,
    )


def sync_fine_tune_trackbars():
    """
    Function: sync_fine_tune_trackbars
    Purpose: Make every trackbar show the current processing value.
    Methods: Temporarily block callbacks and set each control position.
    Creates: Updated GUI trackbar positions.
    """
    global fine_tune_trackbar_update

    fine_tune_trackbar_update = True
    values = (
        ("Brightness", int(BRIGHTNESS)),
        ("Contrast", int(CONTRAST)),
        ("Exposure", int(CAMERA_EXPOSURE)),
        ("Gain", int(CAMERA_GAIN)),
        ("WB Temp", int(CAMERA_WB_TEMP)),
        ("Saturation", int(CAMERA_SATURATION)),
        ("Sharpness", int(CAMERA_SHARPNESS)),
        ("Backlight", int(CAMERA_BACKLIGHT)),
        ("Dark", int(DARK_THRESHOLD)),
        ("Light", int(LIGHT_THRESHOLD)),
        ("Area min", int(area_min)),
        ("Area max", int(area_max)),
        ("Aspect min x100", int(round(ASPECT_MIN * 100))),
        ("Aspect max x100", int(round(ASPECT_MAX * 100))),
        ("Canny low", int(CANNY_LOW)),
        ("Canny high", int(CANNY_HIGH)),
        ("Dilation", int(DILATION_ITERATIONS)),
    )
    for name, value in values:
        cv2.setTrackbarPos(name, FINE_TUNE_WINDOW, value)
    fine_tune_trackbar_update = False


def create_fine_tune_window():
    """
    Function: create_fine_tune_window
    Purpose: Open the mask window and create processing trackbars.
    Methods: Create an OpenCV window and integer-valued controls.
    Creates: Fine Tune Masks window and its trackbars.
    """
    global fine_tune_trackbar_update

    fine_tune_trackbar_update = True
    cv2.namedWindow(FINE_TUNE_WINDOW, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(FINE_TUNE_WINDOW, on_mask_mouse)
    controls = (
        ("Brightness", 255),
        ("Contrast", 255),
        ("Exposure", 50),
        ("Gain", 255),
        ("WB Temp", 7500),
        ("Saturation", 255),
        ("Sharpness", 255),
        ("Backlight", 1),
        ("Dark", 255),
        ("Light", 255),
        ("Area min", AREA_ABS_MAX),
        ("Area max", AREA_ABS_MAX),
        ("Aspect min x100", 1000),
        ("Aspect max x100", 1000),
        ("Canny low", 255),
        ("Canny high", 255),
        ("Dilation", 5),
    )
    for name, maximum in controls:
        cv2.createTrackbar(
            name,
            FINE_TUNE_WINDOW,
            0,
            maximum,
            fine_tune_trackbar_changed,
        )
    fine_tune_trackbar_update = False
    sync_fine_tune_trackbars()


def apply_fine_tune_selection(frame, rect):
    """
    Function: apply_fine_tune_selection
    Purpose: Derive processing limits from a selected plate rectangle.
    Methods: Apply ten-percent area and aspect margins and robust five
             and ninety-five percent grayscale extrema.
    Creates: Updated thresholds, ranges, and synchronized trackbars.
    """
    global DARK_THRESHOLD, LIGHT_THRESHOLD
    global ASPECT_MIN, ASPECT_MAX, area_min, area_max

    if frame is None or rect is None:
        return
    x, y, w, h = rect
    if w < 5 or h < 5:
        print("[fine-tune] Selection is too small.")
        return
    x2 = min(frame.shape[1], x + w)
    y2 = min(frame.shape[0], y + h)
    roi = frame[max(0, y):y2, max(0, x):x2]
    if roi.size == 0:
        return

    selected_area = w * h
    selected_aspect = w / max(h, 1)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    darkest = float(np.percentile(gray, 5))
    lightest = float(np.percentile(gray, 95))
    color_range = max(1.0, lightest - darkest)

    area_min = max(AREA_ABS_MIN, int(selected_area * 0.90))
    area_max = min(AREA_ABS_MAX, int(selected_area * 1.10))
    ASPECT_MIN = max(0.1, selected_aspect * 0.90)
    ASPECT_MAX = selected_aspect * 1.10
    DARK_THRESHOLD = int(np.clip(darkest + color_range * 0.10, 0, 255))
    LIGHT_THRESHOLD = int(np.clip(lightest - color_range * 0.10, 0, 255))
    sync_fine_tune_trackbars()

    print(f"[fine-tune] Selected ROI: x={x} y={y} w={w} h={h}")
    print(
        f"[fine-tune] Measured area={selected_area}, "
        f"aspect={selected_aspect:.3f}"
    )
    print(
        f"[fine-tune] Gray percentiles: "
        f"5%={darkest:.1f}, 95%={lightest:.1f}"
    )
    print_fine_tune_settings("Applied settings")


def labelled_mask(mask, label):
    """
    Function: labelled_mask
    Purpose: Prepare one half-size mask tile with a readable label.
    Methods: Resize, convert grayscale to BGR, and draw a dark label bar.
    Creates: A labelled BGR image tile.
    """
    tile = cv2.resize(
        mask,
        None,
        fx=0.5,
        fy=0.5,
        interpolation=cv2.INTER_NEAREST,
    )
    tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tile, (0, 0), (220, 30), (0, 0, 0), -1)
    cv2.putText(
        tile,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return tile


def show_fine_tune_masks(frame):
    """
    Function: show_fine_tune_masks
    Purpose: Show dark, light, edge, and combined masks in a 2x2 grid.
    Methods: Build detector masks, label half-size tiles, and concatenate.
    Creates: Updated Fine Tune Masks window contents.
    """
    dark, light, edges, combined = make_detection_masks(frame)
    top = np.hstack((
        labelled_mask(dark, "Dark mask"),
        labelled_mask(light, "Light mask"),
    ))
    bottom = np.hstack((
        labelled_mask(edges, "Edges"),
        labelled_mask(combined, "Combined mask"),
    ))
    cv2.imshow(FINE_TUNE_WINDOW, np.vstack((top, bottom)))


def print_fine_tune_settings(title="Settings"):
    """
    Function: print_fine_tune_settings
    Purpose: Print copy-ready image-processing settings.
    Methods: Format all adjustable constants and ranges.
    Creates: Terminal output only.
    """
    print(f"[fine-tune] {title}:")

    print(f"BRIGHTNESS = {BRIGHTNESS}")
    print(f"CONTRAST = {CONTRAST}")
    print(f"CAMERA_EXPOSURE = {CAMERA_EXPOSURE}")
    print(f"CAMERA_GAIN = {CAMERA_GAIN}")
    print(f"CAMERA_WB_TEMP = {CAMERA_WB_TEMP}")
    print(f"CAMERA_SATURATION = {CAMERA_SATURATION}")
    print(f"CAMERA_SHARPNESS = {CAMERA_SHARPNESS}")
    print(f"CAMERA_BACKLIGHT = {CAMERA_BACKLIGHT}")

    print(f"DARK_THRESHOLD = {DARK_THRESHOLD}")
    print(f"LIGHT_THRESHOLD = {LIGHT_THRESHOLD}")
    print(f"ASPECT_MIN = {ASPECT_MIN:.3f}")
    print(f"ASPECT_MAX = {ASPECT_MAX:.3f}")
    print(f"area_min = {area_min}")
    print(f"area_max = {area_max}")
    print(f"CANNY_LOW = {CANNY_LOW}")
    print(f"CANNY_HIGH = {CANNY_HIGH}")
    print(f"DILATION_ITERATIONS = {DILATION_ITERATIONS}")


def toggle_fine_tune():
    """
    Function: toggle_fine_tune
    Purpose: Enter or leave interactive detector calibration mode.
    Methods: Create or destroy the mask window and print final settings.
    Creates: Updated fine_tune_active state.
    """
    global fine_tune_active, fine_tune_input_mode, fine_tune_input_text

    fine_tune_active = not fine_tune_active
    fine_tune_input_mode = None
    fine_tune_input_text = ""
    if fine_tune_active:
        create_fine_tune_window()
        print("[fine-tune] ON. Drag around a plate in Gate Watcher.")
    else:
        print_fine_tune_settings("Final settings")
        cv2.destroyWindow(FINE_TUNE_WINDOW)
        print("[fine-tune] OFF.")


def start_numeric_input(mode):
    """
    Function: start_numeric_input
    Purpose: Start non-blocking numeric entry over the main video.
    Methods: Record the requested setting name and clear the input text.
    Creates: Updated fine_tune_input_mode and fine_tune_input_text.
    """
    global fine_tune_input_mode, fine_tune_input_text

    fine_tune_input_mode = mode
    fine_tune_input_text = ""


def apply_numeric_input():
    """
    Function: apply_numeric_input
    Purpose: Apply typed threshold, area, or ratio values.
    Methods: Parse one central value with a ten-percent margin or parse
             two explicit range endpoints.
    Creates: Updated processing settings and synchronized trackbars.
    """
    global DARK_THRESHOLD, LIGHT_THRESHOLD
    global ASPECT_MIN, ASPECT_MAX, area_min, area_max
    global fine_tune_input_mode, fine_tune_input_text

    parts = fine_tune_input_text.replace(",", " ").split()
    try:
        values = [float(part) for part in parts]
        if fine_tune_input_mode == "DARK" and len(values) == 1:
            DARK_THRESHOLD = int(np.clip(values[0], 0, 255))
        elif fine_tune_input_mode == "LIGHT" and len(values) == 1:
            LIGHT_THRESHOLD = int(np.clip(values[0], 0, 255))
        elif fine_tune_input_mode == "RATIO" and len(values) == 1:
            ASPECT_MIN = max(0.1, values[0] * 0.90)
            ASPECT_MAX = values[0] * 1.10
        elif fine_tune_input_mode == "RATIO" and len(values) == 2:
            ASPECT_MIN, ASPECT_MAX = sorted(values)
        elif fine_tune_input_mode == "AREA" and len(values) == 1:
            area_min = max(AREA_ABS_MIN, int(values[0] * 0.90))
            area_max = min(AREA_ABS_MAX, int(values[0] * 1.10))
        elif fine_tune_input_mode == "AREA" and len(values) == 2:
            low, high = sorted(int(value) for value in values)
            area_min = max(AREA_ABS_MIN, low)
            area_max = min(AREA_ABS_MAX, high)
        else:
            raise ValueError("wrong number of values")
        sync_fine_tune_trackbars()
        print_fine_tune_settings("Manual setting applied")
    except ValueError:
        print(
            f"[fine-tune] Invalid {fine_tune_input_mode} value: "
            f"{fine_tune_input_text!r}"
        )
    fine_tune_input_mode = None
    fine_tune_input_text = ""


def process_fine_tune_key(key):
    """
    Function: process_fine_tune_key
    Purpose: Process calibration hotkeys and non-blocking numeric input.
    Methods: Handle printable characters, Enter, Backspace, and Escape.
    Creates: Updated input state or processing constants.
    """
    global fine_tune_input_mode, fine_tune_input_text

    if fine_tune_input_mode is not None:
        if key in (10, 13):
            apply_numeric_input()
        elif key in (8, 127):
            fine_tune_input_text = fine_tune_input_text[:-1]
        elif key == 27:
            fine_tune_input_mode = None
            fine_tune_input_text = ""
        elif 32 <= key <= 126:
            char = chr(key)
            if char in "0123456789., -":
                fine_tune_input_text += char
        return True

    if key in (ord("b"), ord("B")):
        start_numeric_input("DARK")
    elif key in (ord("w"), ord("W")):
        start_numeric_input("LIGHT")
    elif key in (ord("r"), ord("R")):
        start_numeric_input("RATIO")
    elif key in (ord("a"), ord("A")):
        start_numeric_input("AREA")
    else:
        return False
    return True


def draw_fine_tune_hud(vis, boxes):
    """
    Function: draw_fine_tune_hud
    Purpose: Draw calibration status, selection, and numeric input.
    Methods: Draw text overlays and the selected rectangle in place.
    Creates: Updated visualization frame.
    """
    if fine_tune_rect is not None:
        x, y, w, h = fine_tune_rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 255), 2)
    lines = [
        "FINE TUNE: D=leave, drag=sample, B/W/R/A=numeric input",
        f"dark={DARK_THRESHOLD} light={LIGHT_THRESHOLD} "
        f"area={area_min}..{area_max}",
        f"aspect={ASPECT_MIN:.2f}..{ASPECT_MAX:.2f} "
        f"canny={CANNY_LOW}..{CANNY_HIGH} boxes={len(boxes)}",
    ]
    if fine_tune_input_mode is not None:
        lines.append(
            f"{fine_tune_input_mode}: {fine_tune_input_text}_"
        )
    for index, line in enumerate(lines):
        y = 24 + index * 25
        cv2.rectangle(vis, (5, y - 20), (900, y + 5), (0, 0, 0), -1)
        cv2.putText(
            vis,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


# ---- Main loop -------------------------------------------------------------
def main():
    """
    Function: main
    Purpose: Run video, candidate detection, OCR, and fine tuning.
    Methods: Read camera frames, process modes, draw both windows, and
             dispatch keyboard and GPIO actions without blocking video.
    Creates: Camera capture, visualization frames, and GUI windows.
    """
    global next_read_ts, scan_mode, show_zones
    global area_min, area_max, fine_tune_last_print

    if HW_CONTROL_ENABLED:
        print(
            "HW controls enabled: GPIO buttons select ENTER and EXIT. "
            "Press 'd' fine-tune, 'v' zones, 's' snapshot, "
            "'a' manual spot, or 'q' quit."
        )
    else:
        print(
            "HW controls disabled: press 'z' ENTER, 'x' EXIT, "
            "'d' fine-tune, 'v' zones, 's' snapshot, "
            "'a' manual spot, or 'q' quit."
        )

    cv2.namedWindow(MAIN_WINDOW)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        print("ERROR: Cannot open USB camera!")
        GPIO.cleanup()
        return
    
    disable_camera_auto_controls()

    cv2.setMouseCallback(MAIN_WINDOW, on_mouse, None)
    next_read_ts = 0.0
    last_spot_ts['enter'] = time.monotonic()
    last_spot_ts['exit'] = time.monotonic()
    brightness = BRIGHTNESS
    contrast = CONTRAST

    previous_enter_pressed = False
    previous_exit_pressed = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue

            cv2.setMouseCallback(MAIN_WINDOW, on_mouse, frame)
            now = time.monotonic()

            if not fine_tune_active and HW_CONTROL_ENABLED:
                enter_pressed, exit_pressed = read_gpio_state()

                if enter_pressed and not previous_enter_pressed:
                    toggle_mode('enter')
                elif exit_pressed and not previous_exit_pressed:
                    toggle_mode('exit')

                previous_enter_pressed = enter_pressed
                previous_exit_pressed = exit_pressed

            boxes = find_plate_candidates(frame)
            best_left, best_right = pick_best_by_side(
                boxes,
                frame.shape[1],
            )
            best_exit = best_right if EXIT_ON_RIGHT else best_left
            best_enter = best_left if EXIT_ON_RIGHT else best_right

            if not fine_tune_active and scan_mode != 'idle':
                if AUTO_IDLE_ENABLED:
                    if scan_mode == 'enter':
                        if best_enter:
                            last_spot_ts['enter'] = now
                        elif (
                            now - last_spot_ts['enter']
                        ) >= MAX_TIME_TO_IDLE:
                            print(
                                "Auto-IDLE: no ENTER spots for "
                                f"{MAX_TIME_TO_IDLE:.1f}s"
                            )
                            toggle_mode('enter')
                    elif scan_mode == 'exit':
                        if best_exit:
                            last_spot_ts['exit'] = now
                        elif (
                            now - last_spot_ts['exit']
                        ) >= MAX_TIME_TO_IDLE:
                            print(
                                "Auto-IDLE: no EXIT spots for "
                                f"{MAX_TIME_TO_IDLE:.1f}s"
                            )
                            toggle_mode('exit')

                if now >= next_read_ts:
                    next_read_ts = now + READ_PERIOD
                    if scan_mode == 'exit' and best_exit:
                        p, conf, _, x_center = ocr_bbox(
                            frame,
                            best_exit,
                            side_hint='exit',
                        )
                        if p:
                            update_streak(aggr_left, p, x_center)
                            add_sample(aggr_left, p, conf, x_center)
                            maybe_finalize(aggr_left, frame.shape[1])
                    if scan_mode == 'enter' and best_enter:
                        p, conf, _, x_center = ocr_bbox(
                            frame,
                            best_enter,
                            side_hint='enter',
                        )
                        if p:
                            update_streak(aggr_right, p, x_center)
                            add_sample(aggr_right, p, conf, x_center)
                            maybe_finalize(aggr_right, frame.shape[1])

            vis = frame.copy()
            if show_zones:
                draw_zones(vis)
            for box in boxes:
                draw_box_with_area(vis, box)

            if fine_tune_active:
                draw_fine_tune_hud(vis, boxes)
                show_fine_tune_masks(frame)
                if now - fine_tune_last_print >= FINE_TUNE_PRINT_PERIOD:
                    fine_tune_last_print = now
                    print(
                        f"[fine-tune] candidates={len(boxes)} "
                        f"dark={DARK_THRESHOLD} light={LIGHT_THRESHOLD}"
                    )
                    print(
                        f"[fine-tune] area={area_min}..{area_max} "
                        f"aspect={ASPECT_MIN:.3f}..{ASPECT_MAX:.3f} "
                        f"boxes={boxes}"
                    )
            else:
                hud = "MODE: "
                if scan_mode == 'idle':
                    hud += "IDLE"
                    color = (200, 200, 200)
                elif scan_mode == 'enter':
                    hud += (
                        "ENTER (Right)" if not EXIT_ON_RIGHT
                        else "ENTER (Left)"
                    )
                    color = (0, 255, 0)
                else:
                    hud += (
                        "EXIT (Left)" if not EXIT_ON_RIGHT
                        else "EXIT (Right)"
                    )
                    color = (0, 255, 255)
                cv2.putText(
                    vis,
                    hud,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(MAIN_WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF

            if not main_keyboard_active:
                key = 255

            if key in (ord('d'), ord('D')):
                toggle_fine_tune()
                continue
            if fine_tune_active and process_fine_tune_key(key):
                continue
            if key == ord('q'):
                print("Exiting")
                break
            if fine_tune_active:
                continue
            if key == ord('s'):
                name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, vis)
                print(f"Frame saved into {name}")
            elif not HW_CONTROL_ENABLED and key == ord('z'):
                toggle_mode('enter')
            elif not HW_CONTROL_ENABLED and key == ord('x'):
                toggle_mode('exit')
            elif key == ord('v'):
                show_zones = not show_zones
                print(
                    f"Zone overlay: "
                    f"{'ON' if show_zones else 'OFF'}"
                )
            elif key == ord('a'):
                manual_select_and_process(frame, boxes)
            elif key == ord('['):
                area_min = max(AREA_ABS_MIN, area_min - AREA_STEP)
                area_max = max(area_min, area_max - AREA_STEP)
                print(f"Area range: {area_min}-{area_max}")
            elif key == ord(']'):
                area_min = min(AREA_ABS_MAX, area_min + AREA_STEP)
                area_max = min(
                    AREA_ABS_MAX,
                    max(area_min, area_max + AREA_STEP),
                )
                print(f"Area range: {area_min}-{area_max}")
            elif key == ord(','):
                brightness = set_brightness(brightness - 5)
            elif key == ord('.'):
                brightness = set_brightness(brightness + 5)
            elif key == ord(';'):
                contrast = set_contrast(contrast - 5)
            elif key == ord("'"):
                contrast = set_contrast(contrast + 5)
    except KeyboardInterrupt:
        print("\nInterrupted by Ctrl+C. Exiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()


if __name__ == "__main__":
    main()