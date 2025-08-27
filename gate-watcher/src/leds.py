import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
# import servo # servo.py


'''
Entry pinout
13 = red
6 = green
5 = blue
19 = entry button

Exit pintout
9 = red
0 = green
11 = blue
10 = exit button
'''
entry_leds = [13, 6, 5]
exit_leds = [9, 0, 11]
entry_button_pin = 19
exit_button_pin = 10

# Setup button pins
GPIO.setup(entry_button_pin, GPIO.IN)
GPIO.setup(exit_button_pin, GPIO.IN)

# Setup LED pins
for pin in entry_leds:
    GPIO.setup(pin, GPIO.OUT)
for pin in exit_leds:
    GPIO.setup(pin, GPIO.OUT)

try:
    while True:
        if GPIO.input(entry_button_pin) == GPIO.HIGH:
            servo.set_gate(0, False)  # Open gate 0 (Entry gate)
            print("Entry Button Pressed")
            for pin in entry_leds:
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(pin, GPIO.LOW)
            servo.set_gate(0, True)   # Close gate 0 (Entry gate)

        if GPIO.input(exit_button_pin) == GPIO.HIGH:
            servo.set_gate(1, False)
            print("Exit Button Pressed")
            for pin in exit_leds:
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(1)
                GPIO.output(pin, GPIO.LOW)
            servo.set_gate(1, True)
    
except KeyboardInterrupt:
    print("Done")

finally:
    GPIO.cleanup()