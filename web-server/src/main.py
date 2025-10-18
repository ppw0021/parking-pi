from flask import Flask, jsonify, request, render_template
import sqlite3
import time
import re

# Constant for time (seconds)
entryGracePeriod = 10
exitGracePeriod = 300

# This creates the database if it does not exist
initCon = sqlite3.connect("parkinglot.db")
initCur = initCon.cursor()

# This erases the database
initCur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = initCur.fetchall()
for tableName in tables:
    initCur.execute(f"DROP TABLE IF EXISTS {tableName[0]}")
    print("Reset")
initCon.commit()

# This creates a fresh table with camelCase columns
initCur.execute("""
CREATE TABLE IF NOT EXISTS parkingLot (
    plate TEXT,
    timeIn INTEGER,
    paidToTime INTEGER
)
""")

initCon.commit()
initCur.close()
initCon.close()

app = Flask(__name__)

# Global variable for hourly rate
hourly_rate = 1000

# Get hourly rate
@app.route("/hourly-rate")
def get_hourly_rate():
    return jsonify({"hourlyRate": hourly_rate}), 200

# Update hourly rate
@app.route("/hourly-rate", methods=["POST"])
def update_hourly_rate():
    global hourly_rate
    try:
        new_rate = request.json.get("hourlyRate")
        if not new_rate or not isinstance(new_rate, (int, float)) or new_rate <= 0:
            return jsonify({"error": "Invalid hourly rate"}), 400
            
        hourly_rate = float(new_rate)
        return jsonify({"message": "Hourly rate updated successfully", "hourlyRate": hourly_rate}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Customer panel
@app.route("/")
def root():
    return render_template("index.html")

# Check if a plate exists
@app.get("/check_plate/<plate>")
def checkPlate(plate):
    plate = plate.lower()
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()

        # Get timeIn and paidToTime for the plate
        cur.execute("SELECT timeIn, paidToTime FROM parkingLot WHERE plate = ?", (plate,))
        result = cur.fetchone()

        if not result:
            return jsonify({
                "plate": plate,
                "exists": False,
                "totalParkedSeconds": 0,
                "paid": False
            }), 200
        
        timeNow = int(time.time())

        timeIn, paidToTime = result
        timeOwed = timeNow - paidToTime

        isPaid = True
        if timeOwed > 0:
            isPaid = False

        return jsonify({
            "plate": plate,
            "exists": True,
            "timeOwed": timeOwed,
            "paid": isPaid
        }), 200

    except Exception as e:
        print(f"Error checking plate {plate}: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        cur.close()
        con.close()

# Pay
@app.route("/pay/<plate>")
def pay(plate):
    plate = plate.lower()
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()

        # Check if plate exists
        cur.execute("SELECT 1 FROM parkingLot WHERE plate = ?", (plate,))
        if not cur.fetchone():
            return jsonify({"error": "Plate not found"}), 404

        # Update paidToTime
        paidToTime = int(time.time()) + exitGracePeriod
        cur.execute(
            "UPDATE parkingLot SET paidToTime = ? WHERE plate = ?",
            (paidToTime, plate)
        )
        con.commit()
        return jsonify({"plate": plate, "paidToTime": paidToTime, "message": "Payment successful"}), 200

    except Exception as e:
        print(f"Error paying for plate {plate}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        con.close()

# Admin panel
@app.route("/admin")
def admin():
    return render_template("admin_index.html")

# Get all vehicles (admin only)
@app.route("/admin/vehicles")
def get_vehicles():
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()
        
        cur.execute("SELECT plate, timeIn, paidToTime FROM parkingLot")
        vehicles = cur.fetchall()
        
        current_time = int(time.time())
        vehicle_list = []
        
        for plate, time_in, paid_to_time in vehicles:
            vehicle_list.append({
                "plate": plate,
                "timeIn": time_in,
                "paidToTime": paid_to_time,
                "isPaid": paid_to_time > current_time
            })
            
        return jsonify(vehicle_list), 200
        
    except Exception as e:
        print(f"Error fetching vehicles: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        con.close()

# Called by gate-watcher

# Called by gate-watcher 
# Adds plate and time to database
# (210: added | 211: already exists | 213: error or invalid plate format)
@app.get("/enter/<plate>")
def enter(plate):
    plate = plate.lower()
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()
        # Validate plate format: 3 letters + 3 digits
        valid = re.fullmatch(r"[A-Za-z]{3}\d{3}", plate)
        if not valid:
            raise Exception("invalid plate format")

        timeIn = int(time.time())
        paidToTime = timeIn + entryGracePeriod

        # If plate exists
        cur.execute("SELECT 1 FROM parkingLot WHERE plate = ?", (plate,))
        existing = cur.fetchone()
        if existing:
            return jsonify({
                "message": "already exists"
            }), 211

        # Insert new entry
        cur.execute(
            "INSERT INTO parkingLot (plate, timeIn, paidToTime) VALUES (?, ?, ?)",
            (plate, timeIn, paidToTime)
        )
        con.commit()

        return jsonify({
            "message": f"{plate} added"
        }), 210

    except Exception as e:
        print(f"{plate} : {e}")
        return jsonify({
            "message": f"{plate} error",
            "error": f"{e}"
        }), 213
    finally:
        cur.close()
        con.close()

# Called by gate-watcher 
# Checks if plate has paid
# (210: car paid, exit allowed | 211: car not paid, exit, not allowed | 212: plate not found | 213: error)
@app.get("/exit/<plate>")
def exitLot(plate):
    plate = plate.lower()
    try:
        con = sqlite3.connect("parkinglot.db")
        cur = con.cursor()

        cur.execute("SELECT * FROM parkingLot WHERE plate = ?", (plate,))
        result = cur.fetchone()
        if not result:
            return jsonify({
                "message": f"{plate} not found"
            }), 212
        
        _, _, paidToTime = result
        timeNow = int(time.time())
        exitPermitted = paidToTime > timeNow

        if exitPermitted:
            # Delete the entry from the database
            cur.execute("DELETE FROM parkingLot WHERE plate = ?", (plate,))
            con.commit()
            return jsonify({
                "message": f"{plate} paid up, exit permitted"
            }), 210
        else:
            return jsonify({
                "message": f"{plate} not paid up, exit not permitted"
            }), 211

    except Exception as e:
        print(f"error: {e}")
        return jsonify({
            "message": f"{plate} error",
            "error": f"{e}"
        }), 213
    finally:
        cur.close()
        con.close()

parkingSpots = {str(i): False for i in range(16)}  # False = free, True = taken

@app.post("/update_spots")
def updateSpots():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    for spot in data:
        if "id" in spot and "taken" in spot:
            spotId = str(spot["id"])
            if spotId in parkingSpots:
                parkingSpots[spotId] = bool(spot["taken"])
            else:
                return jsonify({"error": f"Invalid spot id: {spotId}"}), 400
        else:
            return jsonify({"error": "Each spot must have 'id' and 'taken'"}), 400

    return jsonify({
        "message": "Spots updated successfully",
        "spots": parkingSpots
    }), 200

@app.get("/spots")
def getSpots():
    return jsonify(parkingSpots)

app.run(host="0.0.0.0", port=5000, debug=True)
