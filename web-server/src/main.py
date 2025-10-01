# These are like tools we need to build our parking website.
# Flask helps us make the website, and other tools help with time, files, and data.
from flask import Flask, request, render_template_string, send_from_directory, url_for
from flask_socketio import SocketIO
from datetime import datetime
import os
import json

# ------------------------------------------------------------------------------
# This is the start of our parking monitor server.
# It will:
# - Receive pictures and parking info from small computers (like Raspberry Pi).
# - Show a webpage with a table of parked cars and their pictures.
# - Use WebSocket to refresh the page automatically when new data comes in.
# ------------------------------------------------------------------------------

# We create the main website app here.
# '__name__' tells Flask where to find things like templates and files.
# 'static_folder' is where we keep images and other files.
app = Flask(__name__, static_folder='./static')

# This adds WebSocket support to our app.
# It lets the webpage update itself without needing to press refresh.
socketio = SocketIO(app)

# This is a dictionary (like a magic notebook) where we store info about each parking spot.
parking_slots = {}

# This tells us how long cars are allowed to park in different types of spots.
# For example, short = 5 minutes, medium = 90 minutes, long = 240 minutes.
max_duration = {
    "short": 5,
    "medium": 90,
    "long": 240
}

# ------------------------------------------------------------------------------
# Debugging the Websocket connection:
# ------------------------------------------------------------------------------

@socketio.on('connect')
def handle_connect():
    print("Client connected through WebSocket")


# ------------------------------------------------------------------------------
# This part lets us show pictures from the 'static/images' folder.
# If someone asks for '/static/images/car.jpg', we find and send that file.
# ------------------------------------------------------------------------------

@app.route('/static/images/<filename>')
def serve_image(filename):
    # This sends the requested image file to the browser.
    return send_from_directory('./static/images', filename)

# ------------------------------------------------------------------------------
# This part is for Raspberry Pi devices to send us parking data and pictures.
# When they POST (send) data to '/upload/pi1', we save it.
# ------------------------------------------------------------------------------

@app.route('/upload/<pi_id>', methods=['POST'])
def upload(pi_id):
    # Get the image file from the request
    image = request.files.get('image')
    # Get the parking data in text form (JSON)
    data_json = request.form.get('data')

    # If there's an image, save it in the images folder with the Pi's name
    if image:
        image_path = os.path.join('./static/images', f"{pi_id}.jpg")
        image.save(image_path)

    # If there's parking data, turn it into a dictionary and save it
    if data_json:
        try:
            data = json.loads(data_json)  # Convert text to dictionary
            slot_id = data.get("slot_id")  # Get the parking spot ID
            if slot_id is not None:
                parking_slots[slot_id] = data  # Save the data in our notebook
        except json.JSONDecodeError:
            pass  # If the data is broken, ignore it

    # Tell all browsers to update their page
    socketio.emit('update')
    return "Upload received", 200  # Send a message back to the Pi

# ------------------------------------------------------------------------------
# This is the main webpage.
# It shows a table of parked cars and their pictures.
# ------------------------------------------------------------------------------

@app.route('/')
def index():
    now = datetime.now()  # Get the current time
    rows = []  # This will be the list of rows in our table

    # Go through each parking spot and prepare its info for the table
    for slot_id in sorted(parking_slots.keys()):
        slot = parking_slots[slot_id]
        slot_type = slot.get("slot_type", "")
        status = slot.get("status", "")
        license_plate = slot.get("license_plate", "")
        timestamp_str = slot.get("timestamp", "")
        timestamp = datetime.fromisoformat(timestamp_str) if timestamp_str else None

        time_parked = ""        # How long the car has been parked
        duration_minutes = ""   # In minutes
        row_color = ""          # Color of the row (normal, warning, or alert)

        # If the spot is occupied and we have a time, calculate how long it's been
        if status == "occupied" and timestamp:
            delta = now - timestamp
            duration_minutes = max(int(delta.total_seconds() // 60), 0)
            time_parked = f"{duration_minutes} min"

            # Check if the car has stayed too long
            allowed = max_duration.get(slot_type, 0)
            if allowed > 0:
                if duration_minutes > allowed:
                    row_color = "lightcoral"  # Red = too long
                elif duration_minutes >= allowed * 0.9:
                    row_color = "lightyellow"  # Yellow = almost too long

        # Show the type of spot and its time limit
        slot_type_display = f"{slot_type}[{max_duration.get(slot_type, '')}min]" if slot_type in max_duration else slot_type

        # Add this row to the table
        rows.append({
            "slot_type": slot_type_display,
            "slot_id": slot_id,
            "license_plate": license_plate if license_plate else "",
            "timestamp": timestamp_str if status == "occupied" else "",
            "time_parked": time_parked,
            "row_color": row_color
        })

    # Show the webpage with the table and pictures
    return render_template_string(HTML_TEMPLATE, rows=rows)

# ------------------------------------------------------------------------------
# This is the HTML code for the webpage.
# It shows the table and pictures, and updates automatically when new data comes.
# ------------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Parking Monitor</title>

    <style>
        /* This is the style for the webpage */
        body {
            font-family: Arial, sans-serif;
            margin: 10px;
        }
        .container {
            display: flex;
            flex-direction: row;
            gap: 20px;
        }
        .table-container {
            flex: 1;
        }
        .images-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 10px;
            align-items: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            border: 1px solid #aaa;
            padding: 8px;
            text-align: center;
        }
        tr.highlight-yellow {
            background-color: lightyellow;
        }
        tr.highlight-red {
            background-color: lightcoral;
        }
        @media (max-width: 1280px) {
            .container {
                flex-direction: column;
            }
            .images-container {
                flex-direction: column;
            }
        }
    </style>

    <!-- Correct Socket.IO script loading -->
    https://cdn.socket.io/4.5.4/socket.io.min.js

    <!-- JavaScript logic for WebSocket and auto-refresh -->
    <script>
        document.addEventListener("DOMContentLoaded", () => {
            // Connect to the server via WebSocket
            const socket = io();

            // Refresh the page when 'update' event is received
            socket.on('update', () => {
                console.log("Received update event from server");
                location.reload();
            });

            // Refresh the page every 5 minutes
            setInterval(() => {
                console.log("Refreshing page due to 5-minute interval");
                location.reload();
            }, 300000); // 300000 ms = 5 minutes
        });
    </script>
</head>
<body>
    <h2>Parking Monitor</h2>
    <div class="container">
        <div class="table-container">
            <table>
                <tr>
                    <th>Type</th>
                    <th>Slot #</th>
                    <th>License Plate</th>
                    <th>Time Parked</th>
                    <th>Duration</th>
                </tr>
                {% for row in rows %}
                <tr class="{% if row.row_color == 'lightyellow' %}highlight-yellow{% elif row.row_color == 'lightcoral' %}highlight-red{% endif %}">
                    <td>{{ row.slot_type }}</td>
                    <td>{{ row.slot_id }}</td>
                    <td>{{ row.license_plate }}</td>
                    <td>{{ row.timestamp }}</td>
                    <td>{{ row.time_parked }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        <div class="images-container">
            <img src="{{ url_for('static', filename='images/pi1.jpg') }}" alt="Pi1 Image" width="100%">
            <img src="{{ url_for('static', filename='images/pi2.jpg') }}" alt="Pi2 Image" width="100%">
        </div>
    </div>
</body>
</html>
"""

# ------------------------------------------------------------------------------
# This is the part that starts the server.
# It runs only if we launch this file directly.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    # Make sure the images folder exists, or create it
    os.makedirs('./static/images', exist_ok=True)
    # Start the server so people can visit the webpage
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
