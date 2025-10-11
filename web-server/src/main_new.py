from flask import Flask
import sqlite3

# con = sqlite3.connect("parkinglot.db")

# cur = con.cursor()

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
