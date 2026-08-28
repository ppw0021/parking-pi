// Exit page: send the plate to GET /exit/<plate>. On 210 the web server
// removes the car and pulses the exit boom gate open (see gate.py).

const plateInput = document.getElementById("plateInput");
const exitBtn = document.getElementById("exitBtn");
const messageBox = document.getElementById("messageBox");

const PLATE_RE = /^[A-Za-z]{3}\d{3}$/;

function showMessage(message, type = "info") {
    messageBox.className = `info-box ${type}`;
    messageBox.innerHTML = `<p>${message}</p>`;
}

async function requestExit() {
    const plate = plateInput.value.trim();

    if (!PLATE_RE.test(plate)) {
        showMessage("Enter a plate as 3 letters then 3 numbers, e.g. ABC123.", "error");
        return;
    }

    exitBtn.disabled = true;
    showMessage("Contacting the gate…");

    try {
        const res = await fetch(`/exit/${encodeURIComponent(plate.toLowerCase())}`);
        const data = await res.json().catch(() => ({}));

        if (res.status === 210) {
            showMessage(
                `Paid up — the exit gate is opening. Safe travels, ${plate.toUpperCase()}!`,
                "success"
            );
            plateInput.value = "";
        } else if (res.status === 211) {
            showMessage(
                `${plate.toUpperCase()} is not paid up. Please visit the ` +
                `<a href="/pay">Pay</a> page first, then come back.`,
                "error"
            );
        } else if (res.status === 212) {
            showMessage(
                `${plate.toUpperCase()} isn't parked here. Did you enter at the gate?`,
                "error"
            );
        } else {
            showMessage(
                `Could not open the gate: ${data.error || "unknown error"}.`,
                "error"
            );
        }
    } catch (err) {
        console.error(err);
        showMessage("Network error — please try again.", "error");
    } finally {
        exitBtn.disabled = false;
    }
}

exitBtn.addEventListener("click", requestExit);
plateInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") requestExit();
});
