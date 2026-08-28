'''
GateWatcher v1.0 (rewrite)

Camera frame layout:
  LEFT  half  = EXIT lane   - a car waiting to leave the car park
  RIGHT half  = ENTER lane  - a car waiting to come in

Run cal.py first to draw a search zone for each lane (saved to
regions.json).  Nothing is scanned on a timer - the window is just a
live preview until the driver asks to be let through:

  ENTER : press the button on ENTRY_BUTTON_PIN   (keyboard: 'e')
  EXIT  : press the button on EXIT_BUTTON_PIN    (keyboard: 'x')

...and only then is that lane read - a blocking job:

  1. poll frames until the white ~2:1 rectangle (within 45 deg of
     level) shows up in the zone;
  2. deskew it and OCR just that crop, the way up cal.py's rot180
     flag says (the 180 deg flip is a fallback only);
  3. repeat 1-2 until READS_PER_COMMIT samples are gathered (or
     ACQUIRE_TIMEOUT_SEC elapses);
  4. take the most common sample and accept it only if it is exactly
     3 letters + 3 digits, else refuse the request (red blink).

With no regions.json each sample instead OCRs the whole frame and
splits ENTER / EXIT down the middle.

The plate found in that lane at that instant is committed:

  DEMO_MODE = True   -> skip the web server completely.  Print
                        "enter <plate>" / "exit <plate>" and open the
                        gate anyway.
  DEMO_MODE = False  -> ask the web server (unchanged HTTP contract):
                          GET  <URL>/enter/<plate>
                          GET  <URL>/exit/<plate>
                        reply 210 -> open gate (green LED, servo open,
                                     wait GATE_DELAY, servo close)
                        reply 211 -> deny        (steady red LED)
                        reply 212 / 213 / other / network error
                                  -> error       (red LED blinks 3x @ 1 Hz)

The recognised text is sent to the server lower-cased and space-stripped.
Hardware (LEDs, buttons, servos, GPIO pins) and the server contract are
unchanged from the previous version.

Keys:  e = enter request   x = exit request   s = snapshot
       d = toggle debug (show the image tesseract sees)
       r = reload regions.json               q = quit
'''

# ==================== DEPLOY: EDIT THESE FIRST ========================
# The two settings that change per machine / network, kept at the very
# top so they are the first thing you touch on a new install.

# DEMO_MODE = True  -> never contact the web server; open the gate on any
#                      valid plate.  Use this to bench-test the motors
#                      with no server running.
# DEMO_MODE = False -> ask the web server below to allow / deny (normal).
DEMO_MODE = True

# Web server (source of truth).  "/enter/<plate>" / "/exit/<plate>" and
# port 5000 are appended.  Past addresses:
#   http://192.168.1.16    http://10.130.1.206
WEB_PI_IP = "http://10.0.0.2"
URL = f"{WEB_PI_IP}:5000"
HTTP_TIMEOUT = 5.0
# =====================================================================

import json
import os
import time
from collections import Counter
from datetime import datetime

import cv2
import numpy as np
import requests
import pytesseract
from pytesseract import Output

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):   # not on a Pi - run with a no-op stub
    import fake_gpio as GPIO
from leds import LedControl

# ============================ Configuration ===============================
# (DEMO_MODE and WEB_PI_IP live at the very top of this file.)

# Camera.  On a PC the built-in webcam is usually index 0; override with
# the GATE_CAMERA_INDEX env var without editing this file.
CAMERA_INDEX = int(os.environ.get("GATE_CAMERA_INDEX", "1"))
CAMERA_RESOLUTION = (1280, 720)

# Gate servos
SERVO_ENTRY_PIN = 23
SERVO_EXIT_PIN = 24
SERVO_OPEN_ANGLE = {'enter': 70, 'exit': 90}
GATE_DELAY = 5                       # seconds the gate stays open

# LEDs (red, green, blue) per side
ENTRY_LED_PINS = [13, 6, 5]
EXIT_LED_PINS = [9, 0, 11]

# Buttons (read as active-HIGH, matching the existing wiring)
ENTRY_BUTTON_PIN = 19
EXIT_BUTTON_PIN = 10

# OCR regions.  cal.py writes regions.json with a rectangle per lane
# marking where to read text (and whether that lane's plates face away
# from the camera).  If the file is present main.py OCRs only those two
# crops; if not, it falls back to OCRing the whole frame and splitting
# ENTER / EXIT down the middle at LANE_SPLIT.
REGIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "regions.json")
LANE_SPLIT = 0.50                    # fallback lane split (fraction of width)

# Inside each calibrated zone, find the single best WHITE rectangle - the
# number plate.  It is white, roughly 2:1 to 3:1, and sits within
# PLATE_MAX_TILT_DEG of level - the plate's expected orientation is
# horizontal (cal.py's "rot180" only picks which way up OCR is tried
# first, it does not change that the plate lies level in the zone).  A
# candidate must sit inside PLATE_ASPECT +/- PLATE_ASPECT_TOL, cover
# between PLATE_MIN_FILL and PLATE_MAX_FILL of the zone, be a near-solid
# rectangle (PLATE_MIN_EXTENT / PLATE_MIN_SOLIDITY) and have its long
# edge within PLATE_MAX_TILT_DEG of horizontal.  Of the survivors the
# biggest / most rectangular / most 2.5:1 wins.  Run with GATE_DEBUG=1
# (or press 'd') to see the white mask and, on the console, why
# candidates were rejected.
PLATE_ASPECT = float(os.environ.get("GATE_PLATE_ASPECT", "2.5"))               # long / short (US ~2.0, NZ ~2.8)
PLATE_ASPECT_TOL = float(os.environ.get("GATE_PLATE_ASPECT_TOL", "0.50"))      # +/- fraction -> ~[1.25, 3.75]
PLATE_MIN_FILL = float(os.environ.get("GATE_PLATE_MIN_FILL", "0.02"))          # min rect / zone area
PLATE_MAX_FILL = float(os.environ.get("GATE_PLATE_MAX_FILL", "0.98"))          # max rect / zone area (tight zone = plate fills it)
PLATE_MIN_EXTENT = float(os.environ.get("GATE_PLATE_MIN_EXTENT", "0.55"))      # contour / rect area
PLATE_MIN_SOLIDITY = float(os.environ.get("GATE_PLATE_MIN_SOLIDITY", "0.80"))  # contour / hull area
# Max degrees the plate's long edge may be off the expected (level)
# orientation.  45 is the hard ceiling: past 45 deg a 2:1 rectangle is
# indistinguishable from the same rectangle stood on its short end.
PLATE_MAX_TILT_DEG = min(float(os.environ.get("GATE_PLATE_MAX_TILT_DEG", "45")), 45.0)
PLATE_WHITE_MAX_SAT = int(os.environ.get("GATE_PLATE_WHITE_MAX_SAT", "120"))   # HSV S ceiling for "white" (255 disables)
PLATE_WHITE_VAL_MIN = int(os.environ.get("GATE_PLATE_WHITE_VAL_MIN", "100"))   # HSV V floor: never call a dark zone white
PLATE_WARP_H = 200
PLATE_WARP_W = int(round(PLATE_WARP_H * PLATE_ASPECT))

#   GATE_PLATE_WHITELIST    - characters tesseract may output for a plate
#   GATE_PLATE_PSMS         - page-seg modes to try on the plate (best conf wins)
#   GATE_FALLBACK_TESS_CFG  - config for whole-frame OCR when no regions.json
#   GATE_MIN_OCR_CONF       - drop words below this confidence (0-100)
#   GATE_DEBUG=1            - show the image tesseract actually sees
PLATE_WHITELIST = os.environ.get(
    "GATE_PLATE_WHITELIST", "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PLATE_PSMS = [int(p) for p in
              os.environ.get("GATE_PLATE_PSMS", "7,8,13").split(",") if p.strip()]
FALLBACK_TESS_CFG = os.environ.get("GATE_FALLBACK_TESS_CFG", "--oem 1 --psm 11")
MIN_OCR_CONF = float(os.environ.get("GATE_MIN_OCR_CONF", "30"))
DEBUG = os.environ.get("GATE_DEBUG", "0") not in ("0", "")

# Nudge an OCR result toward the local plate layout: L = letter, D = digit
# ('LLLDDD' == the AAA999 contract).  Empty string disables it.  A raw read
# is only rewritten when the nudge makes it match the layout exactly.
PLATE_FORMAT = os.environ.get("GATE_PLATE_FORMAT", "LLLDDD").upper()

# Timing
PLATE_TTL = 4.0                      # how long a just-read plate lingers on the overlay
COMMIT_COOLDOWN = 3.0               # ignore repeat requests on the same lane

# A button / key press kicks off a BLOCKING read: poll frames until the
# white 2:1 rectangle appears, OCR that crop, and repeat until
# READS_PER_COMMIT samples are gathered (or ACQUIRE_TIMEOUT_SEC elapses).
# The most common sample wins, and is accepted only if it is exactly
# 3 letters + 3 digits; otherwise the request is refused.
READS_PER_COMMIT = int(os.environ.get("GATE_READS_PER_COMMIT", "8"))
ACQUIRE_TIMEOUT_SEC = float(os.environ.get("GATE_ACQUIRE_TIMEOUT_SEC", "10.0"))

MAIN_WINDOW = "Gate Watcher"

# ============================ GPIO setup ================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for _pin in ENTRY_LED_PINS + EXIT_LED_PINS + [SERVO_ENTRY_PIN, SERVO_EXIT_PIN]:
    GPIO.setup(_pin, GPIO.OUT)
GPIO.setup(ENTRY_BUTTON_PIN, GPIO.IN)
GPIO.setup(EXIT_BUTTON_PIN, GPIO.IN)

led = LedControl(ENTRY_LED_PINS, EXIT_LED_PINS)

# Runtime state
last_plate = {'enter': ('', 0.0), 'exit': ('', 0.0)}   # side -> (plate, ts), overlay only
last_commit_ts = {'enter': 0.0, 'exit': 0.0}
prev_button = {'enter': False, 'exit': False}

last_ocr_input = None     # debug: last thresholded image handed to tesseract
last_quads = {}           # side -> 4x2 plate corners in frame coords (overlay)


# ============================ Hardware ==================================

def button_pin(side):
    return ENTRY_BUTTON_PIN if side == 'enter' else EXIT_BUTTON_PIN


def set_gate(side, close):
    """Drive one gate servo open or closed (50 Hz PWM, brief dwell)."""
    pin = SERVO_ENTRY_PIN if side == 'enter' else SERVO_EXIT_PIN
    angle = 0 if close else SERVO_OPEN_ANGLE[side]
    duty = 2.5 + (angle / 18.0)
    pwm = GPIO.PWM(pin, 50)
    pwm.start(0)
    try:
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.7)
    finally:
        pwm.stop()
    print(f"  gate {side}: {'closed' if close else f'open {angle}deg'}")


def open_gate(side):
    """Success sequence: green on, gate open, wait, gate closed, green off."""
    led.green_on(side)
    set_gate(side, close=False)
    time.sleep(GATE_DELAY)
    set_gate(side, close=True)
    led.green_off(side)


def clear_side(side):
    """Turn every LED on one side off."""
    led.red_off(side)
    led.green_off(side)
    led.blue_off(side)


def button_rising_edge(side):
    """True once when a lane's button goes from released to pressed."""
    now_high = (GPIO.input(button_pin(side)) == GPIO.HIGH)
    fired = now_high and not prev_button[side]
    prev_button[side] = now_high
    return fired


# ============================ Vision ===================================

SIDE_COLOR = {'enter': (80, 220, 80), 'exit': (0, 165, 255)}   # BGR


def load_regions():
    """Read regions.json -> {side: {'box': (x, y, w, h), 'rot180': bool}}.
    Missing / unreadable file -> {} (main.py then OCRs the whole frame)."""
    try:
        with open(REGIONS_FILE) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    out = {}
    for side in ('enter', 'exit'):
        r = data.get(side) or {}
        box = r.get('box')
        if isinstance(box, list) and len(box) == 4:
            out[side] = {'box': tuple(int(v) for v in box),
                         'rot180': bool(r.get('rot180'))}
    return out


REGIONS = load_regions()


def to_ocr_gray(img):
    """Plain grayscale handed straight to tesseract - it does its own
    binarisation, which beats a hand-rolled threshold under uneven light."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def zone_of(frame, side):
    """Return (zone_bgr, (x0, y0)) for a calibrated side, or (None, None)
    if its box lands off-frame."""
    x, y, w, h = REGIONS[side]['box']
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None, None
    return frame[y1:y2, x1:x2], (x1, y1)


_D2L = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S',
        '6': 'G', '7': 'T', '8': 'B', '9': 'G'}
_L2D = {'O': '0', 'Q': '0', 'D': '0', 'I': '1', 'J': '1', 'L': '1', 'Z': '2',
        'E': '3', 'A': '4', 'S': '5', 'G': '6', 'T': '7', 'B': '8'}


def _coerce_one(s, fmt):
    out = []
    for ch, kind in zip(s, fmt):
        if kind == 'D' and not ch.isdigit():
            out.append(_L2D.get(ch, ch))
        elif kind == 'L' and not ch.isalpha():
            out.append(_D2L.get(ch, ch))
        else:
            out.append(ch)
    return ''.join(out)


def _matches(s, fmt):
    return len(s) == len(fmt) and all(
        (kind == 'L' and ch.isalpha()) or (kind == 'D' and ch.isdigit())
        for ch, kind in zip(s, fmt))


def coerce_format(raw):
    """Return raw rewritten to PLATE_FORMAT if a digit<->letter swap (and
    at most one stray edge character) makes it fit exactly; else raw.
    Among the options, the one needing the fewest swaps wins."""
    if not PLATE_FORMAT:
        return raw
    s = ''.join(ch for ch in raw.upper() if ch.isalnum())
    n = len(PLATE_FORMAT)
    if len(s) == n:
        cands = [s]
    elif len(s) == n + 1:                     # drop one stray edge character
        cands = [s[:-1], s[1:]]
    else:
        return raw
    best = None
    for cand in cands:
        fixed = _coerce_one(cand, PLATE_FORMAT)
        if not _matches(fixed, PLATE_FORMAT):
            continue
        swaps = sum(a != b for a, b in zip(cand, fixed))
        if best is None or swaps < best[1]:
            best = (fixed, swaps)
    return best[0] if best else raw


def _orient_quad(pts):
    """Order 4 corners as TL, TR, BR, BL of the deskewed *landscape*
    view.  Of the two long edges, the one higher in the image (smaller
    y) is the top; within each long edge the left-hand point (smaller x)
    comes first.  This gives a fixed [TL, TR, BR, BL] winding that
    matches the warp destination, so the plate is never mirrored and -
    unlike a sum/diff ordering - never comes out upside-down for a plate
    that is merely tilted.  (Top-vs-bottom is genuinely ambiguous only
    within a few degrees of 45 deg tilt; ocr_plate + cal.py's rot180
    settle up-vs-down.)"""
    pts = np.asarray(pts, dtype="float32").reshape(-1, 2)
    c = pts.mean(axis=0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    e = [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]
    if e[0] + e[2] < e[1] + e[3]:          # make (p0,p1) & (p2,p3) the long edges
        pts = np.roll(pts, -1, axis=0)
    edge_a, edge_b = [pts[0], pts[1]], [pts[2], pts[3]]
    if edge_a[0][1] + edge_a[1][1] > edge_b[0][1] + edge_b[1][1]:
        edge_a, edge_b = edge_b, edge_a   # edge_a is the top edge (smaller y)
    tl, tr = sorted(edge_a, key=lambda p: p[0])
    bl, br = sorted(edge_b, key=lambda p: p[0])
    return np.array([tl, tr, br, bl], dtype="float32")


def _white_mask(zone):
    """Binary mask of the bright, not-too-saturated (white) pixels in a
    zone, with the black characters and any border nicks sealed into one
    slab.  Threshold = Otsu on a CLAHE-equalised value channel, but never
    below PLATE_WHITE_VAL_MIN so a dark zone can't turn all-white.  The
    saturation ceiling drops obviously coloured regions (set
    GATE_PLATE_WHITE_MAX_SAT=255 to disable it for a pure B/W rig)."""
    hsv = cv2.cvtColor(zone, cv2.COLOR_BGR2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    val = cv2.createCLAHE(2.0, (8, 8)).apply(val)
    otsu, _ = cv2.threshold(val, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    v_thr = max(otsu, float(PLATE_WHITE_VAL_MIN))
    mask = ((val >= v_thr) & (sat <= PLATE_WHITE_MAX_SAT)).astype(np.uint8) * 255
    # close first (bridge the black glyphs / border gaps), then a small
    # open to shave speckle without eating a thin plate border
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=2)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    return mask


def _quad_from_contour(c):
    """A 4-point quad for a contour: its polygon approximation when that
    collapses to four convex corners (a real rectangle, even skewed by
    perspective), else the contour's min-area rectangle."""
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")
    return cv2.boxPoints(cv2.minAreaRect(c)).astype("float32")


def _rect_tilt_deg(box):
    """Signed angle of a rotated rectangle's LONG edge from horizontal
    (the expected plate orientation), folded to (-90, 90].  ~0 for a
    level plate, ~+/-90 for an upright one; find_plate rejects anything
    whose magnitude exceeds PLATE_MAX_TILT_DEG.  Computed from the corner
    points, so it does not depend on the OpenCV-version minAreaRect angle
    convention."""
    box = np.asarray(box, dtype="float32")
    e0, e1 = box[1] - box[0], box[2] - box[1]
    long_e = e0 if np.hypot(*e0) >= np.hypot(*e1) else e1
    return (np.degrees(np.arctan2(long_e[1], long_e[0])) + 90.0) % 180.0 - 90.0


_last_find_diag = ""    # dedupe the DEBUG "why no plate" line


def find_plate(zone):
    """Find the best white ~2.5:1 rectangle (the number plate) in a zone
    and rectify it.  Return (rectified_bgr, quad_in_zone_coords, mask), or
    (None, None, mask) when nothing passes the size / aspect /
    rectangularity / tilt gates.  With GATE_DEBUG on, logs which gate the
    candidates fell at (deduped)."""
    global _last_find_diag
    h, w = zone.shape[:2]
    zone_area = float(w * h)
    mask = _white_mask(zone)

    lo = PLATE_ASPECT * (1.0 - PLATE_ASPECT_TOL)
    hi = PLATE_ASPECT * (1.0 + PLATE_ASPECT_TOL)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    rejects = Counter()
    best_c, best_score = None, 0.0
    for c in contours:
        c_area = cv2.contourArea(c)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
        rect_area = rw * rh
        if c_area < PLATE_MIN_FILL * zone_area or min(rw, rh) < 12:
            rejects['too small'] += 1
            continue
        if not (PLATE_MIN_FILL <= rect_area / zone_area <= PLATE_MAX_FILL):
            rejects['fill'] += 1
            continue
        ratio = max(rw, rh) / min(rw, rh)
        if not (lo <= ratio <= hi):
            rejects['aspect'] += 1
            continue
        extent = c_area / rect_area
        if extent < PLATE_MIN_EXTENT:                # ragged blob, not a slab
            rejects['extent'] += 1
            continue
        hull_area = cv2.contourArea(cv2.convexHull(c))
        if hull_area <= 0 or c_area / hull_area < PLATE_MIN_SOLIDITY:
            rejects['solidity'] += 1
            continue
        if abs(_rect_tilt_deg(cv2.boxPoints(((cx, cy), (rw, rh), ang)))) > PLATE_MAX_TILT_DEG:
            rejects['tilt'] += 1
            continue
        # of the survivors prefer the biggest, most rectangular, most 2.5:1
        score = (rect_area * extent
                 * (1.0 - min(abs(ratio - PLATE_ASPECT) / PLATE_ASPECT, 1.0)))
        if score > best_score:
            best_c, best_score = c, score

    if best_c is None:
        if DEBUG:
            diag = (f"{len(contours)} contours, "
                    + (", ".join(f"{k} x{v}" for k, v in rejects.most_common())
                       or "none over the size floor"))
            if diag != _last_find_diag:
                _last_find_diag = diag
                print(f"[find_plate] no plate - {diag}")
        return None, None, mask

    _last_find_diag = ""
    quad = _orient_quad(_quad_from_contour(best_c))
    dst = np.array([[0, 0], [PLATE_WARP_W - 1, 0],
                    [PLATE_WARP_W - 1, PLATE_WARP_H - 1],
                    [0, PLATE_WARP_H - 1]], dtype="float32")
    warp = cv2.warpPerspective(zone, cv2.getPerspectiveTransform(quad, dst),
                               (PLATE_WARP_W, PLATE_WARP_H))
    return warp, quad, mask


def _plate_cfg(psm):
    return (f"--oem 1 --psm {psm} "
            f"-c tessedit_char_whitelist={PLATE_WHITELIST}")


def _read_view(view):
    """Best PLATE_PSMS read of one binarised plate image, as coerce_format
    text ('' if nothing recognised).  A psm whose read snaps onto
    PLATE_FORMAT beats one that does not; ties break on mean tesseract
    confidence.  MIN_OCR_CONF is deliberately not applied - a correct but
    low-confidence read must still be able to win."""
    best_text, best_key = "", (False, -1.0)
    for psm in PLATE_PSMS:
        data = pytesseract.image_to_data(
            view, config=_plate_cfg(psm), output_type=Output.DICT)
        words, confs = [], []
        for word, conf in zip(data['text'], data['conf']):
            try:
                c = float(conf)
            except (TypeError, ValueError):
                continue
            if word.strip() and c >= 0:
                words.append(word.strip())
                confs.append(c)
        if not words:
            continue
        fitted = coerce_format("".join(words))
        key = (bool(PLATE_FORMAT) and _matches(fitted, PLATE_FORMAT),
               sum(confs) / len(confs))
        if key > best_key:
            best_text, best_key = fitted, key
    return best_text


def ocr_plate(plate_bgr, rot180=False):
    """OCR a rectified plate.  cal.py's rot180 flag is authoritative for
    which way up the plate sits, so that orientation is read and
    returned - the two orientations are NOT compared by OCR confidence,
    which is what occasionally handed back an upside-down plate.  The
    180 deg flip is consulted only when the calibrated orientation does
    not produce a layout-valid plate (e.g. a steep deskew came out
    inverted), and it is accepted only if it itself matches PLATE_FORMAT.
    Returns (text, binarised_image_used)."""
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] < 160:
        s = 200.0 / gray.shape[0]
        gray = cv2.resize(gray, None, fx=s, fy=s,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # a white quiet-zone around the text helps the line-based psm modes
    th = cv2.copyMakeBorder(th, 24, 24, 40, 40, cv2.BORDER_CONSTANT, value=255)

    calib = cv2.rotate(th, cv2.ROTATE_180) if rot180 else th
    text = _read_view(calib)
    if not PLATE_FORMAT or _matches(text, PLATE_FORMAT):
        return text, calib

    flip = th if rot180 else cv2.rotate(th, cv2.ROTATE_180)
    text_flip = _read_view(flip)
    if _matches(text_flip, PLATE_FORMAT):
        return text_flip, flip
    return text, calib


def _set_debug_strip(images):
    """Stack the per-side OCR inputs side by side for the debug window."""
    global last_ocr_input
    grays = [im if im.ndim == 2 else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
             for im in images if im is not None]
    if not grays:
        return
    hmax = max(g.shape[0] for g in grays)
    last_ocr_input = cv2.hconcat(
        [cv2.copyMakeBorder(g, 0, hmax - g.shape[0], 0, 12,
                            cv2.BORDER_CONSTANT, value=0) for g in grays]
    )


def is_valid_plate(s):
    """The plate contract: exactly 3 letters followed by 3 digits."""
    s = (s or "").replace(" ", "")
    return len(s) == 6 and s[:3].isalpha() and s[3:].isdigit()


def find_and_read(frame, side):
    """One shot at one calibrated lane on one frame.  Returns:
      None      - no white 2:1 rectangle in the zone (keep polling)
      ''        - rectangle found but nothing legible
      '<text>'  - rectangle found; best OCR of that crop (coerced)
    Refreshes last_quads / the debug strip as a side effect."""
    if side not in REGIONS:
        return ""
    zone, origin = zone_of(frame, side)
    if zone is None:
        return None
    plate, quad, mask = find_plate(zone)
    if quad is None:                          # rectangle not there yet
        last_quads.pop(side, None)
        _set_debug_strip([to_ocr_gray(zone), mask])   # zone | white mask
        return None
    last_quads[side] = quad + np.asarray(origin, dtype="float32")
    text, vis = ocr_plate(plate, REGIONS[side]['rot180'])
    _set_debug_strip([mask, vis])                     # white mask | OCR input
    return coerce_format(text) if text else ""


def read_fallback(frame, side):
    """No regions.json: OCR the whole frame upright and upside-down and
    keep the words sitting on this lane's half.  Returns the joined text
    ('' for none)."""
    global last_ocr_input
    width = frame.shape[1]
    last_ocr_input = to_ocr_gray(frame)
    words = []
    for rotated in (False, True):
        img = cv2.rotate(frame, cv2.ROTATE_180) if rotated else frame
        data = pytesseract.image_to_data(
            to_ocr_gray(img), config=FALLBACK_TESS_CFG, output_type=Output.DICT
        )
        for i, word in enumerate(data['text']):
            text = word.strip()
            try:
                conf = float(data['conf'][i])
            except (TypeError, ValueError):
                continue
            if not text or conf < MIN_OCR_CONF:
                continue
            xc = data['left'][i] + data['width'][i] / 2.0
            if rotated:
                xc = width - xc
            lane = 'exit' if xc < width * LANE_SPLIT else 'enter'
            if lane == side:
                words.append(text)
    return coerce_format(" ".join(words)) if words else ""


def _flush(cap, n=4):
    """Drop a few buffered frames so the next read is current."""
    for _ in range(n):
        cap.read()


def acquire_plate(cap, side):
    """Blocking.  Poll frames until the white 2:1 rectangle shows up in
    the zone, OCR that crop, and keep going until READS_PER_COMMIT
    samples are collected (or ACQUIRE_TIMEOUT_SEC elapses).  Return the
    most common sample if it is exactly 3 letters + 3 digits, else ''."""
    reads = []
    deadline = time.monotonic() + ACQUIRE_TIMEOUT_SEC
    while len(reads) < READS_PER_COMMIT and time.monotonic() < deadline:
        _flush(cap, 2)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if not REGIONS:
            reads.append(read_fallback(frame, side))
            continue
        got = find_and_read(frame, side)
        if got is None:                      # rectangle not in view yet - wait
            continue
        reads.append(got)                    # a sample ('' if found but unreadable)

    votes = Counter(r for r in reads if r)   # blank reads don't get a vote
    if not votes:
        why = "no rectangle found" if not reads else "rectangle never readable"
        print(f"[{side}] read failed - {why} ({len(reads)} samples)")
        return ""
    tally = ", ".join(f"{v}x{c}" for v, c in votes.most_common())
    winner, _ = votes.most_common(1)[0]
    print(f"[{side}] {len(reads)} reads -> [{tally}] -> '{winner}'")
    if is_valid_plate(winner):
        return winner
    print(f"[{side}] most common read is not 3 letters + 3 digits - refused")
    return ""


def draw_overlay(frame, frame_width):
    """Return a copy of the frame with the OCR regions (or the fallback
    lane split) and the current text for each lane drawn on."""
    vis = frame.copy()
    if REGIONS:
        for side, r in REGIONS.items():
            x, y, w, h = r['box']
            col = SIDE_COLOR[side]
            cv2.rectangle(vis, (x, y), (x + w, y + h), col, 1)
            tag = side.upper() + (" rot180" if r['rot180'] else "")
            cv2.putText(vis, tag, (x, max(14, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        # highlight the one white rectangle found in each zone
        for side, quad in last_quads.items():
            pts = quad.astype(np.int32)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(vis, "PLATE?", (pts[0][0], max(14, pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                        cv2.LINE_AA)
    else:
        split = int(frame_width * LANE_SPLIT)
        cv2.line(vis, (split, 0), (split, frame.shape[0]), (0, 255, 255), 1)

    labels = {
        'exit': ("EXIT", SIDE_COLOR['exit'], (10, 34)),
        'enter': ("ENTER", SIDE_COLOR['enter'], (frame_width - 300, 34)),
    }
    now = time.monotonic()
    for side, (text, color, anchor) in labels.items():
        plate, ts = last_plate[side]
        fresh = plate and (now - ts <= PLATE_TTL)
        cv2.putText(vis, f"{text}: {plate if fresh else '---'}", anchor,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return vis


# ============================ Request handling =========================

def commit(side, cap):
    """Act on a driver's request for one lane (button or key).  Runs the
    blocking find-rectangle -> OCR -> retry sequence (acquire_plate); on
    a valid 3+3 plate, open / deny / error as before, otherwise refuse."""
    now = time.monotonic()
    if now - last_commit_ts[side] < COMMIT_COOLDOWN:
        return
    last_commit_ts[side] = now

    clear_side(side)
    led.blue_on(side)                            # busy: finding + reading plate
    plate = acquire_plate(cap, side)
    if not plate:
        led.blue_off(side)
        print(f"[{side}] request refused - no agreed 3+3 plate from {READS_PER_COMMIT} reads")
        led.blink_red(side, times=3, freq_hz=1.0)
        return

    last_plate[side] = (plate, time.monotonic())   # show it on the overlay
    print(f"{side} {plate}")          # the record of who entered / exited

    if DEMO_MODE:
        led.blue_off(side)
        open_gate(side)
        print(f"[{side}] DEMO: gate opened for {plate}")
        return

    uri = f"{URL}/{side}/{plate.lower().replace(' ', '')}"
    try:
        code = requests.get(uri, timeout=HTTP_TIMEOUT).status_code
    except requests.exceptions.RequestException as exc:
        led.blue_off(side)
        led.blink_red(side, times=3, freq_hz=1.0)
        print(f"[{side}] server unreachable: {exc}")
        return

    led.blue_off(side)
    print(f"[{side}] GET {uri} -> {code}")
    if code == 210:
        open_gate(side)
    elif code == 211:
        led.red_on(side)
        print(f"[{side}] denied ({code})")
    else:                             # 212, 213, or anything unexpected
        led.blink_red(side, times=3, freq_hz=1.0)
        print(f"[{side}] error ({code})")


# ============================ Main loop ================================

def main():
    global DEBUG, REGIONS
    print(__doc__)
    cap = None
    for idx in dict.fromkeys([CAMERA_INDEX, 0]):   # try configured index, then 0
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            print(f"camera: opened index {idx}")
            break
        cap.release()
    if cap is None or not cap.isOpened():
        print("ERROR: cannot open camera (try GATE_CAMERA_INDEX=<n>)")
        GPIO.cleanup()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])

    cv2.namedWindow(MAIN_WINDOW)
    led.all_off()
    for side in ('enter', 'exit'):
        prev_button[side] = (GPIO.input(button_pin(side)) == GPIO.HIGH)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
                continue
            frame_width = frame.shape[1]

            # The camera just previews - no scanning happens here.  A
            # lane is only read when its button / key is pressed below.
            vis = draw_overlay(frame, frame_width)
            if DEBUG and last_ocr_input is not None:
                cv2.imshow("ocr input", last_ocr_input)
            cv2.imshow(MAIN_WINDOW, vis)
            key = cv2.waitKey(1) & 0xFF

            if button_rising_edge('enter') or key in (ord('e'), ord('E')):
                commit('enter', cap)          # blocks: find rectangle, read, retry
            if button_rising_edge('exit') or key in (ord('x'), ord('X')):
                commit('exit', cap)
            if key in (ord('s'), ord('S')):
                name = datetime.now().strftime("gate_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(name, frame)
                print(f"saved {name}")
            if key in (ord('d'), ord('D')):
                DEBUG = not DEBUG
                print(f"debug view {'on' if DEBUG else 'off'}")
                if not DEBUG:
                    cv2.destroyWindow("ocr input")
            if key in (ord('r'), ord('R')):
                REGIONS = load_regions()
                print(f"regions reloaded: {sorted(REGIONS) or 'none (full-frame)'}")
            if key in (ord('q'), ord('Q')):
                break
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        led.all_off()
        GPIO.cleanup()


if __name__ == "__main__":
    main()
