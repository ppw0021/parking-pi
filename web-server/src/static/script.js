const plateInput = document.getElementById("plateInput");
const checkBtn = document.getElementById("checkBtn");
const messageBox = document.getElementById("messageBox");
const infoBox = document.getElementById("infoBox");
const timeParked = document.getElementById("timeParked");
const totalDue = document.getElementById("totalDue");
const payBtn = document.getElementById("payBtn");
const successMessage = document.getElementById("successMessage");
const newPaymentBtn = document.getElementById("newPaymentBtn");
const paymentForm = document.getElementById("paymentForm");

let parkingRate = 999;

// --- Helper function to show messages ---
function showMessage(message, type = "info") {
    messageBox.classList.remove("hidden");
    messageBox.style.backgroundColor = type === "error" ? "#ffdddd" : "#eef6ff";
    messageBox.innerHTML = `<p>${message}</p>`;
}

function hideMessage() {
    messageBox.classList.add("hidden");
}

function hideInfoBox() {
    infoBox.classList.add("hidden");
}
// --- Check button ---
checkBtn.addEventListener("click", async () => {
    const plate = plateInput.value.trim();

    // If input is empty
    if (!plate) {
        showMessage("Please enter a number plate.", "error");
        hideInfoBox();
        return;
    }

    try {
        // Call the Flask endpoint
        const response = await fetch(`/check_plate/${plate}`);
        if (!response.ok) {
            throw new Error("Failed to check plate.");
        }

        const data = await response.json();

        if (data.exists) {
            // Plate exists
            const totalSeconds = data.timeOwed;
            if (data.paid) {
                // Car is paid up, no payment needed
                console.log(`Paid up ${totalSeconds}`)
                const minutesPaid = 0 - Math.floor((totalSeconds % 3600) / 60);
                showMessage(`Plate ${plate} is paid up for the next ${minutesPaid} minute${minutesPaid !== 1 ? 's' : ''}. No payment required.`);
                plateInput.value = ""; // clear input after checking
            } else {
                hideMessage();
                // Show time parked and placeholder total due
                const hours = Math.round(totalSeconds / 3600);
                const minutes = Math.ceil((totalSeconds % 3600) / 60);
                timeParked.textContent = `${hours} hour${hours !== 1 ? 's' : ''} ${minutes} min${minutes !== 1 ? 's' : ''}`;
                console.log((Math.round(hours * parkingRate) + (minutes/60 * parkingRate)));
                totalDue.textContent = `${((Math.round(hours * parkingRate) + (minutes/60 * parkingRate)).toFixed(2))}`; // Rounds to 2 decimal places
                infoBox.style.backgroundColor = "#eef6ff"; // info color
                infoBox.classList.remove("hidden");
            }

        } else {
            // Plate does not exist
            showMessage(`Plate ${plate} not found in database.`, "error");
            hideInfoBox();
        }

    } catch (err) {
        console.error(err);
        showMessage("Error checking plate. See console for details.", "error");
        hideInfoBox();
    }
});

// Periodically fetch parking spot status and update UI
function updateParkingSpots() {
    fetch('/spots')
        .then(response => response.json())
        .then(data => {
            // data is an object: { '1': false, '2': true, ... }
            Object.entries(data).forEach(([id, taken]) => {
                const spot = document.querySelector(`.spot[data-id="${id}"]`);
                if (spot) {
                    if (taken) {
                        spot.classList.add('taken');
                        spot.classList.remove('available');
                    } else {
                        spot.classList.add('available');
                        spot.classList.remove('taken');
                    }
                }
            });
        })
        .catch(err => {
            // Optionally handle error
            // console.error('Failed to fetch spots:', err);
        });
}

// Poll every 5 seconds
setInterval(updateParkingSpots, 1000);
setInterval(updateParkingRate, 4000)

function updateParkingRate() {
    fetch('/hourly-rate')
        .then(response => response.json())
        .then(data => {
            console.log(data.hourlyRate)
            parkingRate = data.hourlyRate
        })
}
// Initial call on page load
window.addEventListener('DOMContentLoaded', updateParkingSpots);

// --- Pay button ---
payBtn.addEventListener("click", async () => {
    const plate = plateInput.value.trim();

    if (!plate) {
        showMessage("No plate to pay for.", "error");
        return;
    }

    try {
        // Send payment request to Flask
        const response = await fetch(`/pay/${plate}`);
        if (!response.ok) {
            throw new Error("Payment failed.");
        }

        const data = await response.json();
        console.log(`Payment successful for plate ${plate}`, data);

        // Hide payment form & show success message
        infoBox.classList.add("hidden");
        messageBox.classList.add("hidden");
        paymentForm.querySelector(".form-group").style.display = "none";
        successMessage.classList.remove("hidden");

        // Optionally, you could display the paidToTime returned
        // e.g., showMessage(`Paid until: ${new Date(data.paidToTime * 1000).toLocaleTimeString()}`);
        plateInput.value = ""; // clear input after checking

    } catch (err) {
        console.error(err);
        showMessage("Error during payment. See console for details.", "error");
    }
});

// --- Pay for another car ---
newPaymentBtn.addEventListener("click", () => {
    // Reset form
    plateInput.value = "";
    infoBox.classList.add("hidden");
    messageBox.classList.add("hidden");
    paymentForm.querySelector(".form-group").style.display = "block";
    successMessage.classList.add("hidden");
});

updateParkingRate()