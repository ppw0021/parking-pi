const plateInput = document.getElementById("plateInput");
const checkBtn = document.getElementById("checkBtn");
const infoBox = document.getElementById("infoBox");
const timeParked = document.getElementById("timeParked");
const totalDue = document.getElementById("totalDue");
const payBtn = document.getElementById("payBtn");
const successMessage = document.getElementById("successMessage");
const newPaymentBtn = document.getElementById("newPaymentBtn");
const paymentForm = document.getElementById("paymentForm");

// --- Check button ---
checkBtn.addEventListener("click", async () => {
    const plate = plateInput.value.trim();
    if (!plate) {
        alert("Please enter a number plate.");
        return;
    }

    // Placeholder: simulate fetching time & total
    console.log(`Fetching parking info for plate: ${plate}`);
    timeParked.textContent = "1 hour 24 mins";
    totalDue.textContent = "4.50";
    infoBox.classList.remove("hidden");
});

// --- Pay button ---
payBtn.addEventListener("click", async () => {
    const plate = plateInput.value.trim();
    console.log(`Sending payment request for plate: ${plate}`);

    // Hide payment form & show success message
    infoBox.classList.add("hidden");
    paymentForm.querySelector(".form-group").style.display = "none";
    successMessage.classList.remove("hidden");
});

// --- Pay for another car ---
newPaymentBtn.addEventListener("click", () => {
    // Reset form
    plateInput.value = "";
    infoBox.classList.add("hidden");
    paymentForm.querySelector(".form-group").style.display = "flex";
    successMessage.classList.add("hidden");
});
