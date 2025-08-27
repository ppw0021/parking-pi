'''
Required for setup
sudo apt update
sudo apt install pigpio python3-pigpio

sudo systemctl enable pigpiod
sudo systemctl start pigpiod


'''
import RPi.GPIO as GPIO
from time import sleep

ENTRY_PIN = 16  # physical pin 16
EXIT_PIN = 18

def set_gate(gate_id: int, close: bool):
    FREQ = 50  # 50 Hz for servo
    PIN = 0
    if gate_id == 0:
        PIN = ENTRY_PIN
    elif gate_id == 1:
        PIN = EXIT_PIN
    else:
        return

    if close:
        # Close
        angle = 0

    else:
        # Open
        angle = 90

    # Safety clamp
    angle = max(0, min(180, angle))

    # Convert angle to duty cycle (approximate for SG90)
    duty = 2.5 + (angle / 18.0)  # maps 0–180° to ~2.5–12.5% duty

    GPIO.setmode(GPIO.BOARD)
    GPIO.setwarnings(False)
    GPIO.setup(PIN, GPIO.OUT)

    pwm = GPIO.PWM(PIN, FREQ)
    pwm.start(0)

    try:
        pwm.ChangeDutyCycle(duty)
        sleep(0.5)  # wait for the servo to reach position
    finally:
        pwm.stop()
        GPIO.cleanup()

    print(f"Moved servo to {angle}° (duty cycle {duty:.2f}%)")


# Example usage:

# Open Gates
# set_gate(0, False)  # Open gate 0 (Entry gate)
# set_gate(1, False)  # Open gate 1 (Exit gate)
# sleep(1)

# #Close Gates
# set_gate(0, True)   # Close gate 0 (Entry gate)
# set_gate(1, True)   # Close gate 1 (Exit gate)