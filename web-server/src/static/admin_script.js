// Get DOM elements
const vehicleTableBody = document.querySelector('#vehicleTable tbody');
const currentRateDisplay = document.getElementById('currentRate');
const newRateInput = document.getElementById('newRateInput');
const updateRateBtn = document.getElementById('updateRateBtn');

// Format timestamp to readable date/time
function formatTimestamp(timestamp) {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
}

// Update vehicle table with data
function updateVehicleTable() {
    fetch('/admin/vehicles')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(vehicles => {
            // Clear existing table rows
            vehicleTableBody.innerHTML = '';

            // Add each vehicle to the table
            vehicles.forEach(vehicle => {
                const row = document.createElement('tr');
                
                // Calculate payment status and style
                const isPaid = vehicle.isPaid;
                const statusClass = isPaid ? 'paid' : 'unpaid';
                const statusText = isPaid ? 'Paid' : 'Unpaid';

                row.innerHTML = `
                    <td>${vehicle.plate.toUpperCase()}</td>
                    <td>${formatTimestamp(vehicle.timeIn)}</td>
                    <td>${formatTimestamp(vehicle.paidToTime)}</td>
                    <td class="${statusClass}">${statusText}</td>
                `;
                
                vehicleTableBody.appendChild(row);
            });
        })
        .catch(error => {
            console.error('Error fetching vehicle data:', error);
            vehicleTableBody.innerHTML = `
                <tr>
                    <td colspan="4" class="error">Error loading vehicle data</td>
                </tr>
            `;
        });
}

// Update parking spots same as main page
function updateParkingSpots() {
    fetch('/spots')
        .then(response => response.json())
        .then(data => {
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
            console.error('Failed to fetch spots:', err);
        });
}

// Add some extra styles for the admin table
const style = document.createElement('style');
style.textContent = `
    #vehicleTable {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
    }

    #vehicleTable th,
    #vehicleTable td {
        padding: 0.5rem;
        text-align: left;
        border-bottom: 1px solid #ddd;
    }

    #vehicleTable th {
        background-color: #0074D9;
        color: white;
    }

    #vehicleTable tr:nth-child(even) {
        background-color: #f8f9fa;
    }

    #vehicleTable .paid {
        color: #2ECC40;
        font-weight: bold;
    }

    #vehicleTable .unpaid {
        color: #FF4136;
        font-weight: bold;
    }

    #vehicleTable .error {
        color: #FF4136;
        text-align: center;
        padding: 1rem;
    }

    @media (max-width: 1200px) {
        #vehicleTable {
            font-size: 0.8em;
        }
    }
`;
document.head.appendChild(style);

// Update hourly rate display
function updateHourlyRateDisplay() {
    fetch('/hourly-rate')
        .then(response => response.json())
        .then(data => {
            currentRateDisplay.textContent = `$${data.hourlyRate.toFixed(2)}`;
        })
        .catch(error => {
            console.error('Error fetching hourly rate:', error);
        });
}

// Handle rate update
updateRateBtn.addEventListener('click', () => {
    const newRate = parseFloat(newRateInput.value);
    if (isNaN(newRate) || newRate <= 0) {
        alert('Please enter a valid rate greater than 0');
        return;
    }

    fetch('/hourly-rate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ hourlyRate: newRate })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        updateHourlyRateDisplay();
        alert('Hourly rate updated successfully');
    })
    .catch(error => {
        console.error('Error updating rate:', error);
        alert('Failed to update rate: ' + error.message);
    });
});

// Initial updates
updateVehicleTable();
updateParkingSpots();
updateHourlyRateDisplay();

// Refresh data periodically
setInterval(updateVehicleTable, 5000);  // every 5 seconds
setInterval(updateParkingSpots, 1000);  // every 1 second