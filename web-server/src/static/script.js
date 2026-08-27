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

let parkingRate = 0;

// --- Formatting helpers ---
function fmtDuration(seconds) {
    seconds = Math.max(0, Math.round(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h && m) return `${h}h ${m}m`;
    if (h) return `${h}h`;
    return `${m}m`;
}

function money(n) {
    return `$${Math.max(0, n).toFixed(2)}`;
}

// --- Message helpers ---
function showMessage(message, type = "info") {
    messageBox.className = `info-box ${type}`;
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

    if (!plate) {
        showMessage("Please enter a number plate.", "error");
        hideInfoBox();
        return;
    }

    try {
        const response = await fetch(`/check_plate/${encodeURIComponent(plate)}`);
        if (!response.ok) throw new Error("Failed to check plate.");
        const data = await response.json();

        if (!data.exists) {
            showMessage(`Plate ${plate.toUpperCase()} was not found.`, "error");
            hideInfoBox();
            return;
        }

        // timeOwed > 0  => unpaid time owed, in seconds
        // timeOwed <= 0 => paid ahead by that many seconds
        if (data.paid || data.timeOwed <= 0) {
            const remaining = fmtDuration(-data.timeOwed);
            showMessage(
                `Plate ${plate.toUpperCase()} is paid up (${remaining} remaining). No payment required.`,
                "success"
            );
            hideInfoBox();
            plateInput.value = "";
            return;
        }

        hideMessage();
        timeParked.textContent = fmtDuration(data.timeOwed);
        totalDue.textContent = money((data.timeOwed / 3600) * parkingRate);
        infoBox.classList.remove("hidden");
    } catch (err) {
        console.error(err);
        showMessage("Error checking plate. See console for details.", "error");
        hideInfoBox();
    }
});

// --- Pay button ---
payBtn.addEventListener("click", async () => {
    const plate = plateInput.value.trim();
    if (!plate) {
        showMessage("No plate to pay for.", "error");
        return;
    }

    try {
        const response = await fetch(`/pay/${encodeURIComponent(plate)}`);
        if (!response.ok) throw new Error("Payment failed.");
        await response.json();

        hideInfoBox();
        hideMessage();
        paymentForm.querySelector(".form-group").style.display = "none";
        successMessage.classList.remove("hidden");
        plateInput.value = "";
    } catch (err) {
        console.error(err);
        showMessage("Error during payment. See console for details.", "error");
    }
});

// --- Pay for another car ---
newPaymentBtn.addEventListener("click", () => {
    plateInput.value = "";
    hideInfoBox();
    hideMessage();
    paymentForm.querySelector(".form-group").style.display = "block";
    successMessage.classList.add("hidden");
});

// --- Live parking spots ---
let lastOk = 0;

function markUpdated(fresh) {
    if (fresh) lastOk = Date.now();
    const el = document.getElementById("lastUpdated");
    if (!el) return;
    if (!lastOk) {
        el.textContent = "Waiting for data…";
        return;
    }
    const age = Math.round((Date.now() - lastOk) / 1000);
    el.textContent = age < 5 ? "Updated just now" : `Updated ${age}s ago`;
    el.classList.toggle("stale", age > 10);
}

function updateParkingSpots() {
    fetch("/spots")
        .then((response) => response.json())
        .then((data) => {
            let free = 0;
            Object.entries(data).forEach(([id, taken]) => {
                const spot = document.querySelector(`.spot[data-id="${id}"]`);
                if (!spot) return;
                spot.classList.toggle("taken", !!taken);
                spot.classList.toggle("available", !taken);
                spot.title = `Spot ${Number(id) + 1} — ${taken ? "taken" : "available"}`;
                if (!taken) free++;
            });
            const freeEl = document.getElementById("spotsFree");
            if (freeEl) freeEl.textContent = free;
            const summary = document.getElementById("spotsSummary");
            if (summary) summary.classList.toggle("full", free === 0);
            markUpdated(true);
        })
        .catch((err) => {
            console.error("Failed to fetch spots:", err);
            markUpdated(false);
        });
}

function updateParkingRate() {
    fetch("/hourly-rate")
        .then((response) => response.json())
        .then((data) => {
            parkingRate = data.hourlyRate;
        })
        .catch((err) => console.error("Failed to fetch rate:", err));
}

updateParkingSpots();
updateParkingRate();
setInterval(updateParkingSpots, 1000);
setInterval(updateParkingRate, 4000);
setInterval(() => markUpdated(false), 1000);
