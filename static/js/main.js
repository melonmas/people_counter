document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const totalInEl = document.getElementById('totalIn');
    const totalOutEl = document.getElementById('totalOut');
    const currentInsideEl = document.getElementById('currentInside');

    // Initialize Chart
    const ctx = document.getElementById('visitorChart').getContext('2d');
    const visitorChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Current Visitors',
                data: [],
                borderColor: '#2979ff',
                backgroundColor: 'rgba(41, 121, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Real-time Occupancy Trend',
                    color: '#b3b3b3',
                    font: { family: "'Roboto', sans-serif" }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: '#333' },
                    ticks: { color: '#888' }
                },
                x: {
                    grid: { display: false },
                    ticks: { display: false } // Hide time labels for clean look
                }
            }
        }
    });

    // Helper functions
    function updateStats() {
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                // Update numbers
                totalInEl.innerText = data.total_in;
                totalOutEl.innerText = data.total_out;
                currentInsideEl.innerText = data.current_inside;

                // Update Chart
                updateChart(data.current_inside);
            })
            .catch(err => console.error('Error fetching stats:', err));
    }

    function updateChart(value) {
        const now = new Date();
        const timeLabel = now.getHours() + ':' + now.getMinutes() + ':' + now.getSeconds();

        // Add Data
        visitorChart.data.labels.push(timeLabel);
        visitorChart.data.datasets[0].data.push(value);

        // Keep only last 20 points
        if (visitorChart.data.labels.length > 20) {
            visitorChart.data.labels.shift();
            visitorChart.data.datasets[0].data.shift();
        }

        visitorChart.update();
    }

    // Polling interval (1 second)
    setInterval(updateStats, 1000);
});
