const vehicleTableBody = document.querySelector("#vehicleTable tbody");
const currentRateDisplay = document.getElementById("currentRate");
const newRateInput = document.getElementById("newRateInput");
const updateRateBtn = document.getElementById("updateRateBtn");
const rateMsg = document.getElementById("rateMsg");

let rateInputSeeded = false;

// --- Formatting helpers ---
function formatTimestamp(timestamp) {
    return new Date(timestamp * 1000).toLocaleString();
}

function fmtDuration(seconds) {
    seconds = Math.max(0, Math.round(seconds));
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    if (h && m) return `${h}h ${m}m`;
    if (h) return `${h}h`;
    return `${m}m`;
}

// --- Live "updated Xs ago" indicator ---
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

// --- Vehicle table ---
function updateVehicleTable() {
    fetch("/admin/vehicles")
        .then((response) => {
            if (!response.ok) throw new Error("Network response was not ok");
            return response.json();
        })
        .then((vehicles) => {
            vehicleTableBody.innerHTML = "";

            if (!vehicles.length) {
                vehicleTableBody.innerHTML =
                    '<tr><td colspan="5" class="empty">No vehicles currently parked</td></tr>';
                return;
            }

            const now = Date.now() / 1000;
            vehicles.forEach((vehicle) => {
                const row = document.createElement("tr");
                const statusClass = vehicle.isPaid ? "paid" : "unpaid";
                const statusText = vehicle.isPaid ? "Paid" : "Unpaid";

                row.innerHTML = `
                    <td>${vehicle.plate.toUpperCase()}</td>
                    <td>${formatTimestamp(vehicle.timeIn)}</td>
                    <td>${fmtDuration(now - vehicle.timeIn)}</td>
                    <td>${formatTimestamp(vehicle.paidToTime)}</td>
                    <td class="${statusClass}">${statusText}</td>
                `;
                vehicleTableBody.appendChild(row);
            });
        })
        .catch((error) => {
            console.error("Error fetching vehicle data:", error);
            vehicleTableBody.innerHTML =
                '<tr><td colspan="5" class="empty">Could not load vehicle data</td></tr>';
        });
}

// --- Live parking spots ---
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

// --- Hourly rate ---
function updateHourlyRateDisplay() {
    fetch("/hourly-rate")
        .then((response) => response.json())
        .then((data) => {
            currentRateDisplay.textContent = `$${data.hourlyRate.toFixed(2)}`;
            if (!rateInputSeeded && !newRateInput.value) {
                newRateInput.value = data.hourlyRate;
                rateInputSeeded = true;
            }
        })
        .catch((error) => console.error("Error fetching hourly rate:", error));
}

function showRateMsg(text, isError) {
    rateMsg.textContent = text;
    rateMsg.classList.toggle("stale", !!isError);
    clearTimeout(showRateMsg._t);
    showRateMsg._t = setTimeout(() => {
        rateMsg.textContent = "";
        rateMsg.classList.remove("stale");
    }, 4000);
}

updateRateBtn.addEventListener("click", () => {
    const newRate = parseFloat(newRateInput.value);
    if (isNaN(newRate) || newRate <= 0) {
        showRateMsg("Enter a rate greater than 0.", true);
        return;
    }

    updateRateBtn.disabled = true;
    fetch("/hourly-rate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ hourlyRate: newRate }),
    })
        .then((response) => response.json())
        .then((data) => {
            if (data.error) throw new Error(data.error);
            updateHourlyRateDisplay();
            showRateMsg(`Rate updated to $${newRate.toFixed(2)}/hr.`);
        })
        .catch((error) => {
            console.error("Error updating rate:", error);
            showRateMsg(`Failed to update rate: ${error.message}`, true);
        })
        .finally(() => {
            updateRateBtn.disabled = false;
        });
});

// --- Init + polling ---
updateVehicleTable();
updateParkingSpots();
updateHourlyRateDisplay();
setInterval(updateVehicleTable, 5000);
setInterval(updateParkingSpots, 1000);
setInterval(() => markUpdated(false), 1000);
