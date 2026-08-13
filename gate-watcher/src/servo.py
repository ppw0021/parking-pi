import RPi.GPIO as GPIO
from time import sleep

ENTRY_PIN = 23  # BCM GPIO 23, physical pin 16
EXIT_PIN = 24   # BCM GPIO 24, physical pin 18
FREQ = 50

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(ENTRY_PIN, GPIO.OUT)
GPIO.setup(EXIT_PIN, GPIO.OUT)

_pwm = {
    0: GPIO.PWM(ENTRY_PIN, FREQ),
    1: GPIO.PWM(EXIT_PIN, FREQ),
}
for p in _pwm.values():
    p.start(0)

def set_gate(gate_id: int, close: bool):
    if gate_id not in _pwm:
        return
    angle = 0 if close else 90
    duty = 2.5 + (angle / 18.0)
    _pwm[gate_id].ChangeDutyCycle(duty)
    sleep(0.7)
    _pwm[gate_id].ChangeDutyCycle(0)  # stop jitter
    print(f"Gate {gate_id} {'closed' if close else 'opened'} → {angle}° ({duty:.2f}%)")
