from flask import Flask
import sqlite3
import time

# This creates the database if it does not exist
con = sqlite3.connect("parkinglot.db")
cur = con.cursor()

# This erases the database
cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()
for table_name in tables:
    cur.execute(f"DROP TABLE IF EXISTS {table_name[0]}")
con.commit()

# This creates a 
cur.execute("""
CREATE TABLE IF NOT EXISTS parking_lot (
    plate TEXT,
    time_in INTEGER,
    paid_to_time INTEGER
)
""")
con.commit()

cur.execute("INSERT INTO parking_lot (plate, time_in, paid_to_time) VALUES (?, ?, ?)",
            ("ABC123", 1700000000, 1700003600))
con.commit()

# Query all rows
cur.execute("SELECT * FROM parking_lot")
rows = cur.fetchall()

con.close()

for row in rows:
    print(row)

app = Flask(__name__)

@app.route("/")
def root():
    return "Customer Panel", 509

@app.route("/customer")
def customer():
    return "Customer Panel"

@app.route("/admin")
def admin():
    return "Admin Panel"

@app.get("/enter/<plate>")
def enter(plate):
    # Here, plate will be added to database, car has entered
    fail = False
    if (not fail):
        return f"{plate}", 110
    else:
        return f"{plate}", 111


@app.get("/exit/<plate>")
def exit(plate):
    # Here, code will check if the car is allowed to leave or not 
    paid = True
    error = False
    
    if (error):
        return f"{plate}", 122

    if (paid):
        # Paid
        return f"{plate}", 120
    elif (not paid):
        # Not paid
        return f"{plate}", 121

app.run()
