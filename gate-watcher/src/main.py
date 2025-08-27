import requests
import servo # servo.py
from time import sleep

'''
Car entering:
Use URL:
URL/enter/<plate>

Return options:
status_code will be 110 if added correctly.
status_code will be 111 if failed, e.g. car is already in the carpark

'''

# Replace with your target IP (and include http:// or https://)
url = "http://127.0.0.1:5000"

try:
    plate = "54"
    uri = url + "/exit/" + plate
    response = requests.get(uri, timeout=5)
    print(f"Status code: {response.status_code}")

    if (response.status_code == 201):
        print("Customer has paid, open the gate")
        # customer has paid open the gates, blink LED whatever
    elif (response.status_code == 202):
        print("Customer has not paid, do not open gate")
        # Customer has not paid, blink red LED
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")


# Open Gates
servo.set_gate(0, False)  # Open gate 0 (Entry gate)
servo.set_gate(1, False)  # Open gate 1 (Exit gate)
sleep(3)

#Close Gates
servo.set_gate(0, True)   # Close gate 0 (Entry gate)
servo.set_gate(1, True)   # Close gate 1 (Exit gate)