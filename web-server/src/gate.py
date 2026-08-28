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

Gate pulses run on a background thread so the HTTP response returns
straight away instead of blocking for GATE_DELAY seconds.
"""

import threading
import time

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):   # not on a Pi - no-op stub
    import fake_gpio as GPIO

from leds import LedControl

# ==================== DEPLOY: EDIT THESE FIRST ========================
# Wiring for the machine the gates are attached to.  Mirrors the values
# that used to live in gate-watcher/src/main.py.

SERVO_ENTRY_PIN = 23
SERVO_EXIT_PIN = 24
SERVO_OPEN_ANGLE = {"enter": 70, "exit": 90}
GATE_DELAY = 5                       # seconds the gate stays open

# LEDs (red, green, blue) per side
ENTRY_LED_PINS = [13, 6, 5]
EXIT_LED_PINS = [9, 0, 11]
# =====================================================================

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
for _pin in ENTRY_LED_PINS + EXIT_LED_PINS + [SERVO_ENTRY_PIN, SERVO_EXIT_PIN]:
    GPIO.setup(_pin, GPIO.OUT)

led = LedControl(ENTRY_LED_PINS, EXIT_LED_PINS)
led.all_off()

# One lock per side: while a gate is mid-cycle, extra requests for that
# same side are ignored rather than stacking up on the servo.
_busy = {"enter": threading.Lock(), "exit": threading.Lock()}


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
    print(f"  gate {side}: {'closed' if close else f'open {angle}deg'}")


def _open_sequence(side):
    """Green on, gate open, wait GATE_DELAY, gate closed, green off."""
    if not _busy[side].acquire(blocking=False):
        print(f"[gate] {side} already open - ignoring repeat request")
        return
    try:
        led.green_on(side)
        _set_gate(side, close=False)
        time.sleep(GATE_DELAY)
        _set_gate(side, close=True)
        led.green_off(side)
    finally:
        _busy[side].release()


def open_gate(side):
    """Pulse the given gate ('enter' or 'exit') open then shut, off-thread."""
    threading.Thread(target=_open_sequence, args=(side,), daemon=True).start()


def deny(side):
    """Blink the red LED for a rejected request, off-thread."""
    threading.Thread(
        target=led.blink_red, args=(side,), kwargs={"times": 3, "freq_hz": 1.0},
        daemon=True,
    ).start()
