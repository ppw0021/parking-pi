"""
Boom-gate control for the web server.

This is the servo/LED half of the old gate-watcher, moved into the web
server so the customer web app can open the gates itself (see the Entry
and Exit pages).  The HTTP contract is unchanged - main.py still decides
allow/deny in /enter/<plate> and /exit/<plate>; it just also calls
open_gate() / deny() here on the way out.

Runs anywhere:  on the gate Pi it drives real servos through RPi.GPIO
(install `rpi-lgpio`); on a dev PC RPi.GPIO is missing so it falls back
to fake_gpio and every hardware call just prints what it would have done.

GPIO is claimed lazily, on the first gate request, NOT at import - see
_ensure_init().  Gate pulses run on a background thread so the HTTP
response returns straight away instead of blocking for GATE_DELAY
seconds; any hardware error is printed to the server log rather than
dying silently in that thread.
"""

import threading
import time

try:
    import RPi.GPIO as GPIO
    ON_REAL_GPIO = True
except (ImportError, RuntimeError):   # not on a Pi - no-op stub
    import fake_gpio as GPIO
    ON_REAL_GPIO = False

from leds import LedControl

# ==================== DEPLOY: EDIT THESE FIRST ========================
# Wiring for the machine the gates are attached to.  BCM numbering, same
# as gate-watcher/src/main.py.

SERVO_ENTRY_PIN = 23
SERVO_EXIT_PIN = 24
SERVO_OPEN_ANGLE = {"enter": 70, "exit": 90}
GATE_DELAY = 5                       # seconds the gate stays open

# LEDs (red, green, blue) per side
ENTRY_LED_PINS = [13, 6, 5]
EXIT_LED_PINS = [9, 0, 11]
# =====================================================================

led = LedControl(ENTRY_LED_PINS, EXIT_LED_PINS)

# One lock per side: while a gate is mid-cycle, extra requests for that
# same side are ignored rather than stacking up on the servo.
_busy = {"enter": threading.Lock(), "exit": threading.Lock()}

_init_lock = threading.Lock()
_initialised = False


def _log(msg):
    """print() that always reaches the log, even when stdout is piped."""
    print(msg, flush=True)


def _ensure_init():
    """Claim the GPIO lines. Called on the first gate request, once.

    This is deliberately NOT done at import time.  main.py runs Flask
    with debug=True, so the werkzeug auto-reloader imports this module
    in BOTH the reloader process and the worker process.  rpi-lgpio
    claims each line exclusively, so whichever process calls
    GPIO.setup() second gets an error and that half never drives the
    servo - which looks exactly like "the gates don't open".  Doing the
    setup lazily means only the worker that actually handles a request
    ever touches the pins.
    """
    global _initialised
    with _init_lock:
        if _initialised:
            return
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        for pin in ENTRY_LED_PINS + EXIT_LED_PINS + [SERVO_ENTRY_PIN, SERVO_EXIT_PIN]:
            GPIO.setup(pin, GPIO.OUT)
        led.all_off()
        _initialised = True
        kind = "RPi.GPIO (real hardware)" if ON_REAL_GPIO else \
            "fake_gpio stub - NO real hardware, install rpi-lgpio on the Pi"
        _log(f"[gate] GPIO initialised via {kind}")


def _set_gate(side, close):
    """Drive one gate servo open or closed (50 Hz PWM, brief dwell)."""
    pin = SERVO_ENTRY_PIN if side == "enter" else SERVO_EXIT_PIN
    angle = 0 if close else SERVO_OPEN_ANGLE[side]
    duty = 2.5 + (angle / 18.0)
    pwm = GPIO.PWM(pin, 50)
    pwm.start(0)
    try:
        pwm.ChangeDutyCycle(duty)
        time.sleep(0.7)
    finally:
        pwm.stop()
    _log(f"[gate] {side}: {'closed' if close else f'open {angle}deg'} (BCM {pin})")


def _open_sequence(side):
    """Green on, gate open, wait GATE_DELAY, gate closed, green off."""
    if not _busy[side].acquire(blocking=False):
        _log(f"[gate] {side} already open - ignoring repeat request")
        return
    try:
        _ensure_init()
        _log(f"[gate] opening {side} gate")
        led.green_on(side)
        _set_gate(side, close=False)
        time.sleep(GATE_DELAY)
        _set_gate(side, close=True)
        led.green_off(side)
    except Exception as exc:            # don't let it vanish in the thread
        _log(f"[gate] {side} gate FAILED: {exc!r}")
    finally:
        _busy[side].release()


def _deny_sequence(side):
    try:
        _ensure_init()
        led.blink_red(side, times=3, freq_hz=1.0)
    except Exception as exc:
        _log(f"[gate] deny({side}) FAILED: {exc!r}")


def open_gate(side):
    """Pulse the given gate ('enter' or 'exit') open then shut, off-thread."""
    threading.Thread(target=_open_sequence, args=(side,), daemon=True).start()


def deny(side):
    """Blink the red LED for a rejected request, off-thread."""
    threading.Thread(target=_deny_sequence, args=(side,), daemon=True).start()
