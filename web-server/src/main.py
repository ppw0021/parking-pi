from flask import Flask, jsonify, request, render_template
import sqlite3
import time
import re

# This creates the database if it does not exist
initCon = sqlite3.connect("parkinglot.db")
initCur = initCon.cursor()

# This erases the database
initCur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = initCur.fetchall()
for table_name in tables:
    initCur.execute(f"DROP TABLE IF EXISTS {table_name[0]}")
    print("Reset")
initCon.commit()

# This creates a fresh table
initCur.execute("""
CREATE TABLE IF NOT EXISTS parking_lot (
    plate TEXT,
    time_in INTEGER,
    paid_to_time INTEGER
)
""")

initCon.commit()

# Query all rows
# cur.execute("SELECT * FROM parking_lot")
# rows = cur.fetchall()
# for row in rows:
#     print(row)

initCur.close()
initCon.close()

app = Flask(__name__)

# Customer panel
@app.route("/")
def root():
    # Render the HTML file in templates/
    return render_template("index.html")

@app.route("/ping")
def ping():
    print("Ping received!")  # this will show in your terminal
    return jsonify({"message": "Pong from server!"})

# Admin panel
@app.route("/admin")
def admin():
    return "Admin Panel"

# Called by gate-watcher 
# Adds plate and time to database
# (210: added | 211: aready exists | 213: error or invalid plate format)
@app.get("/enter/<plate>")
def enter(plate):
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()
        # Validate plate format: 3 letters + 3 digits
        valid = re.fullmatch(r"[A-Za-z]{3}\d{3}", plate)
        if not valid:
            raise Exception("invalid plate format")

        timeIn = int(time.time())

        # If plate exists
        cur.execute("SELECT 1 FROM parking_lot WHERE plate = ?", (plate,))
        existing = cur.fetchone()
        if existing:
            # Exists, do not add
            return f"{plate} : already exists", 211

        # Insert new entry
        cur.execute(
            "INSERT INTO parking_lot (plate, time_in, paid_to_time) VALUES (?, ?, ?)",
            (plate, timeIn, timeIn)
        )
        con.commit()

        # Success
        return f"{plate} : added", 210

    except Exception as e:
        # Error or invalid plate format
        print(f"{plate} : {e}")
        return f"{plate} : {e}", 213
    finally:
        cur.close()
        con.close()

# Called by gate-watcher 
# Checks if plate has paid
# (210: car paid, exit allowed | 211: car not paid, exit, not allowed | 212: plate not found | 213: error)
@app.get("/exit/<plate>")
def exit(plate):
    # Here, code will check if the car is allowed to leave or not 
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()

        #Use parameterized query for safety and correctness
        cur.execute("SELECT * FROM parking_lot WHERE plate = ?", (plate,))
        result = cur.fetchone()  # get first match (or None if not found)
        if not result:
            # No result found
            return f"plate : {plate} not found", 212
        
        # Logic for checking if car is paid for

        exitPermitted = False

        if exitPermitted:
            # Car is paid up
            return f"plate : {plate} has paid and is allowed to exit", 210
        else:
            # Car is not paid up
            return f"plate : {plate} has not paid and is not allowed to exit", 211

    except Exception as e:
        # Error has occured
        print(f"error: {e}")
        return f"error: {e}", 213
    finally:
        cur.close()
        con.close()

parking_spots = {str(i): False for i in range(16)}  # False = free, True = taken


@app.post("/update_spots")
def update_spots():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    for spot in data:
        if "id" in spot and "taken" in spot:
            spot_id = str(spot["id"])
            if spot_id in parking_spots:
                parking_spots[spot_id] = bool(spot["taken"])
            else:
                return jsonify({"error": f"Invalid spot id: {spot_id}"}), 400
        else:
            return jsonify({"error": "Each spot must have 'id' and 'taken'"}), 400

    return jsonify({
        "message": "Spots updated successfully",
        "spots": parking_spots
    }), 200


@app.get("/spots")
def get_spots():
    return jsonify(parking_spots)

app.run()