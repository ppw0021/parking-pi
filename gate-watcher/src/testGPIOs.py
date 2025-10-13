#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sense HAT arrows on GPIO inputs.

- Up (Exit): show red upward arrow when GPIO14 is LOW (to GND).
- Down (Enter): show blue downward arrow when GPIO4 is LOW (to GND).
- Both: show both arrows when both pins are LOW.
"""

import time
import RPi.GPIO as GPIO
from sense_hat import SenseHat

# ---------------- Config ----------------------------------------------------
PIN_EXIT = 14     # BCM 14 -> Exit -> Up arrow (red)
PIN_ENTER = 4     # BCM 4  -> Enter -> Down arrow (blue)
DEBOUNCE_SEC = 0.05

# Colors (R, G, B)
RED  = (255, 0, 0)
BLUE = (0, 0, 255)
BLK  = (0, 0, 0)

# ---------------- Arrow patterns (8x8) -------------------------------------
# Legend: 'r' -> RED, 'b' -> BLUE, '0' -> BLACK

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

# ---------------- Helpers ---------------------------------------------------
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


def show_arrow(sense, state):
    """
    Function: show_arrow
    Purpose: Display the proper arrow(s) based on GPIO state.
    Methods: Choose pattern; call sense.set_pixels(pixels).
    Creates: local 'pixels' list of 64 tuples.
    """
    enter_low, exit_low = state
    if enter_low and exit_low:
        pixels = pattern_to_pixels(BOTH)
    elif exit_low:
        pixels = pattern_to_pixels(UP_RED)
    elif enter_low:
        pixels = pattern_to_pixels(DOWN_BLUE)
    else:
        sense.clear()
        return
    sense.set_pixels(pixels)


def read_state():
    """
    Function: read_state
    Purpose: Sample GPIO and return tuple of booleans (enter_low, exit_low).
    Methods: GPIO.input() with pull-ups; LOW means grounded/active.
    Creates: two bools.
    """
    enter_low = (GPIO.input(PIN_ENTER) == GPIO.LOW)
    exit_low = (GPIO.input(PIN_EXIT) == GPIO.LOW)
    return (enter_low, exit_low)


# ---------------- Main ------------------------------------------------------
def main():
    """
    Function: main
    Purpose: Init GPIO and Sense HAT, then update LEDs on state change.
    Methods: Poll every DEBOUNCE_SEC; only redraw on change.
    Creates: 'prev' tuple to detect changes.
    """
    sense = SenseHat()
    sense.clear()

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_ENTER, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(PIN_EXIT,  GPIO.IN, pull_up_down=GPIO.PUD_UP)

    prev = (None, None)
    try:
        while True:
            cur = read_state()
            if cur != prev:
                show_arrow(sense, cur)
                prev = cur
            time.sleep(DEBOUNCE_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        sense.clear()
        GPIO.cleanup()


if __name__ == "__main__":
    main()