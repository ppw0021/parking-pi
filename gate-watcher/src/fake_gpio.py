"""
Drop-in no-op stand-in for RPi.GPIO so gate-watcher can run on a normal PC.

main.py / leds.py fall back to this automatically when the real RPi.GPIO
cannot be imported (i.e. not running on a Raspberry Pi).  Buttons always
read LOW - drive enter/exit from the keyboard ('e' / 'x') instead.  LED and
servo calls just print what real hardware would have done.
"""

BCM = "BCM"
BOARD = "BOARD"
OUT = "OUT"
IN = "IN"
HIGH = 1
LOW = 0
PUD_UP = "PUD_UP"
PUD_DOWN = "PUD_DOWN"

_printed = False


def _note():
    global _printed
    if not _printed:
        _printed = True
        print("[fake_gpio] RPi.GPIO not available - using no-op GPIO stub")


def setmode(mode):
    _note()


def setwarnings(flag):
    pass


def setup(pin, mode, initial=None, pull_up_down=None):
    pass


def input(pin):
    return LOW


def output(pin, value):
    pass


def cleanup(pin=None):
    pass


class PWM:
    def __init__(self, pin, frequency):
        self.pin = pin
        self.frequency = frequency

    def start(self, dutycycle):
        pass

    def ChangeDutyCycle(self, dutycycle):
        pass

    def ChangeFrequency(self, frequency):
        self.frequency = frequency

    def stop(self):
        pass
