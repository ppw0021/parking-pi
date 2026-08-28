// Live parking-spots board. Loaded on every customer page (base.html);
// polls GET /spots once a second and repaints the grid.

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

updateParkingSpots();
setInterval(updateParkingSpots, 1000);
setInterval(() => markUpdated(false), 1000);
