// Entry page: send the plate to GET /enter/<plate>. The web server adds
// the car and, on 210, pulses the entry boom gate open (see gate.py).

const plateInput = document.getElementById("plateInput");
const enterBtn = document.getElementById("enterBtn");
const messageBox = document.getElementById("messageBox");

const PLATE_RE = /^[A-Za-z]{3}\d{3}$/;

function showMessage(message, type = "info") {
    messageBox.className = `info-box ${type}`;
    messageBox.innerHTML = `<p>${message}</p>`;
}

async function requestEntry() {
    const plate = plateInput.value.trim();

    if (!PLATE_RE.test(plate)) {
        showMessage("Enter a plate as 3 letters then 3 numbers, e.g. ABC123.", "error");
        return;
    }

    enterBtn.disabled = true;
    showMessage("Contacting the gate…");

    try {
        const res = await fetch(`/enter/${encodeURIComponent(plate.toLowerCase())}`);
        const data = await res.json().catch(() => ({}));

        if (res.status === 210) {
            showMessage(
                `Welcome, ${plate.toUpperCase()}. The gate is opening — please drive in. ` +
                `When you're ready to leave, use the <a href="/pay">Pay</a> page.`,
                "success"
            );
            plateInput.value = "";
        } else if (res.status === 211) {
            showMessage(
                `${plate.toUpperCase()} is already parked here. Head to the ` +
                `<a href="/pay">Pay</a> page when you're ready to leave.`,
                "error"
            );
        } else {
            showMessage(
                `Could not let you in: ${data.error || "invalid plate"}. Please check the plate and try again.`,
                "error"
            );
        }
    } catch (err) {
        console.error(err);
        showMessage("Network error — please try again.", "error");
    } finally {
        enterBtn.disabled = false;
    }
}

enterBtn.addEventListener("click", requestEntry);
plateInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") requestEntry();
});
