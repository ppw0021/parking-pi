# led_control.py
# -*- coding: utf-8 -*-
"""
Module: led_control
Purpose: Encapsulate LED control (on/off/blink) for entry/exit bays.
Methods: Uses RPi.GPIO to drive digital outputs; exposes simple API:
    - all_off()
    - set_color(side, color, on=True)
    - blue_on/blue_off, green_on/green_off, red_on/red_off
    - blink_red(side, times=3, freq_hz=1.0)
Creates: No global state beyond kept pin mapping.
"""

import time
import RPi.GPIO as GPIO


class LedControl:
    """
    Class: LedControl
    Purpose: Manage three LEDs (red/green/blue) for two sides: 'enter'
             and 'exit'. HIGH = on, LOW = off.
    Methods: GPIO.output for steady states; time.sleep for blinking.
    Creates: self._pins dict with pin numbers for both sides.
    """

    def __init__(self, entry_pins, exit_pins):
        """
        Function: __init__
        Purpose: Store pin mapping; no setup is done here assuming pins
                 are configured as outputs in the caller (main.py).
        Methods: Build dicts mapping color names to BCM pins.
        Creates: self._pins['enter'|'exit']['red'|'green'|'blue'].
        """
        self._pins = {
            'enter': {
                'red': entry_pins[0],
                'green': entry_pins[1],
                'blue': entry_pins[2],
            },
            'exit': {
                'red': exit_pins[0],
                'green': exit_pins[1],
                'blue': exit_pins[2],
            },
        }

    def _pins_for_side(self, side):
        """
        Function: _pins_for_side
        Purpose: Return color->pin mapping for 'enter' or 'exit'.
        Methods: Dict access with basic validation.
        Creates: None.
        """
        side_key = 'enter' if side == 'enter' else 'exit'
        return self._pins[side_key]

    def set_color(self, side, color, on=True):
        """
        Function: set_color
        Purpose: Turn a specific color LED on/off for the given side.
        Methods: GPIO.output(pin, HIGH/LOW).
        Creates: None.
        """
        pins = self._pins_for_side(side)
        pin = pins[color]
        GPIO.output(pin, GPIO.HIGH if on else GPIO.LOW)

    def blue_on(self, side):
        """Turn blue LED on for the given side."""
        self.set_color(side, 'blue', True)

    def blue_off(self, side):
        """Turn blue LED off for the given side."""
        self.set_color(side, 'blue', False)

    def green_on(self, side):
        """Turn green LED on for the given side."""
        self.set_color(side, 'green', True)

    def green_off(self, side):
        """Turn green LED off for the given side."""
        self.set_color(side, 'green', False)

    def red_on(self, side):
        """Turn red LED on for the given side."""
        self.set_color(side, 'red', True)

    def red_off(self, side):
        """Turn red LED off for the given side."""
        self.set_color(side, 'red', False)

    def all_off(self):
        """
        Function: all_off
        Purpose: Switch off all LEDs on both sides.
        Methods: Iterate over sides and colors; GPIO.output(LOW).
        Creates: None.
        """
        for side in ('enter', 'exit'):
            pins = self._pins_for_side(side)
            for pin in pins.values():
                GPIO.output(pin, GPIO.LOW)

    def blink_red(self, side, times=3, freq_hz=1.0):
        """
        Function: blink_red
        Purpose: Blink red LED a fixed number of times at given freq.
        Methods: Toggle with sleep(period/2) between states.
        Creates: None.
        """
        period = 1.0 / max(freq_hz, 0.001)
        for _ in range(int(times)):
            self.red_on(side)
            time.sleep(period / 2.0)
            self.red_off(side)
            time.sleep(period / 2.0)