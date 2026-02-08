let vehicleTypeChart = null;
let stateChart = null;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Set today's date
    document.getElementById('dateInput').value = new Date().toISOString().split('T')[0];
    
    // Load initial data
    loadTotalStatistics();
    loadRecentDetections();
    loadVehicleTypeStats();
    loadStateStats();
});

// Tab switching
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + '-content').classList.add('active');
    
    // Add active class to clicked tab
    event.target.classList.add('active');
}

// Load total statistics
async function loadTotalStatistics() {
    try {
        const response = await fetch('/api/statistics/total');
        const result = await response.json();
        
        if (result.success) {
            document.getElementById('totalDetections').textContent = result.data.total_detections || 0;
            document.getElementById('uniqueVehicles').textContent = result.data.unique_vehicles || 0;
            document.getElementById('statesDetected').textContent = result.data.states_detected || 0;
            document.getElementById('avgQuality').textContent = result.data.avg_quality_score || '0.00';
        }
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Load recent detections
async function loadRecentDetections() {
    const limit = document.getElementById('recentLimit').value;
    showLoading();
    
    try {
        const response = await fetch(`/api/recent-detections?limit=${limit}`);
        const result = await response.json();
        
        if (result.success) {
            displayTable(result.data, 'Recent Detections');
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to load data: ' + error.message);
    }
}

// Load by timeframe
async function loadByTimeframe() {
    const hours = document.getElementById('timeframeHours').value;
    showLoading();
    
    try {
        const response = await fetch(`/api/detections-by-timeframe?hours=${hours}`);
        const result = await response.json();
        
        if (result.success) {
            displayTable(result.data, `Detections from Last ${hours} Hour(s)`);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to load data: ' + error.message);
    }
}

// Load by date
async function loadByDate() {
    const date = document.getElementById('dateInput').value;
    showLoading();
    
    try {
        const response = await fetch(`/api/detections-by-date?date=${date}`);
        const result = await response.json();
        
        if (result.success) {
            displayTable(result.data, `Detections for ${date}`);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to load data: ' + error.message);
    }
}

// Search vehicle
async function searchVehicle() {
    const plate = document.getElementById('searchPlate').value;
    
    if (!plate) {
        alert('Please enter a vehicle number');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`/api/search?plate=${encodeURIComponent(plate)}`);
        const result = await response.json();
        
        if (result.success) {
            displayTable(result.data, `Search Results for "${plate}"`);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to search: ' + error.message);
    }
}

// Load high quality detections
async function loadHighQuality() {
    const threshold = document.getElementById('qualityThreshold').value;
    showLoading();
    
    try {
        const response = await fetch(`/api/high-quality?threshold=${threshold}`);
        const result = await response.json();
        
        if (result.success) {
            displayTable(result.data, `High Quality Detections (>= ${threshold})`);
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to load data: ' + error.message);
    }
}

// Export to CSV
async function exportCSV() {
    try {
        window.location.href = '/api/export/csv';
        showSuccess('Export started. Download will begin shortly.');
    } catch (error) {
        showError('Failed to export: ' + error.message);
    }
}

// Delete old records
async function deleteOldRecords() {
    const days = document.getElementById('deleteDays').value;
    
    if (!confirm(`Are you sure you want to delete records older than ${days} days? This cannot be undone.`)) {
        return;
    }
    
    try {
        const response = await fetch(`/api/delete-old?days=${days}&confirm=true`);
        const result = await response.json();
        
        if (result.success) {
            showSuccess(result.message);
            loadTotalStatistics();
            loadRecentDetections();
        } else {
            showError(result.message);
        }
    } catch (error) {
        showError('Failed to delete: ' + error.message);
    }
}

// Load vehicle type statistics
async function loadVehicleTypeStats() {
    try {
        const response = await fetch('/api/statistics/vehicle-type');
        const result = await response.json();
        
        if (result.success && result.data.length > 0) {
            const labels = result.data.map(item => item.vehicle_type);
            const data = result.data.map(item => item.count);
            
            const ctx = document.getElementById('vehicleTypeChart').getContext('2d');
            
            if (vehicleTypeChart) {
                vehicleTypeChart.destroy();
            }
            
            vehicleTypeChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Number of Detections',
                        data: data,
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.7)',
                            'rgba(118, 75, 162, 0.7)',
                            'rgba(237, 100, 166, 0.7)',
                            'rgba(255, 154, 158, 0.7)',
                            'rgba(250, 208, 196, 0.7)'
                        ],
                        borderColor: [
                            'rgba(102, 126, 234, 1)',
                            'rgba(118, 75, 162, 1)',
                            'rgba(237, 100, 166, 1)',
                            'rgba(255, 154, 158, 1)',
                            'rgba(250, 208, 196, 1)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            ticks: {
                                precision: 0
                            }
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error loading vehicle type stats:', error);
    }
}

// Load state statistics
async function loadStateStats() {
    try {
        const response = await fetch('/api/statistics/state');
        const result = await response.json();
        
        if (result.success && result.data.length > 0) {
            const labels = result.data.map(item => `${item.state_code} - ${item.state_name}`);
            const data = result.data.map(item => item.count);
            
            const ctx = document.getElementById('stateChart').getContext('2d');
            
            if (stateChart) {
                stateChart.destroy();
            }
            
            stateChart = new Chart(ctx, {
                type: 'pie',
                data: {
                    labels: labels,
                    datasets: [{
                        data: data,
                        backgroundColor: [
                            'rgba(102, 126, 234, 0.7)',
                            'rgba(118, 75, 162, 0.7)',
                            'rgba(237, 100, 166, 0.7)',
                            'rgba(255, 154, 158, 0.7)',
                            'rgba(250, 208, 196, 0.7)',
                            'rgba(164, 228, 251, 0.7)',
                            'rgba(134, 239, 172, 0.7)',
                            'rgba(253, 224, 71, 0.7)'
                        ],
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: {
                            position: 'right'
                        }
                    }
                }
            });
        }
    } catch (error) {
        console.error('Error loading state stats:', error);
    }
}

// Display table
function displayTable(data, title) {
    document.getElementById('tableTitle').textContent = title;
    
    if (!data || data.length === 0) {
        document.getElementById('tableContent').innerHTML = '<p style="text-align: center; padding: 40px; color: #666;">No data found</p>';
        return;
    }
    
    let html = '<table><thead><tr>';
    
    // Get headers from first row
    const headers = Object.keys(data[0]);
    headers.forEach(header => {
        html += `<th>${header.replace(/_/g, ' ').toUpperCase()}</th>`;
    });
    
    html += '</tr></thead><tbody>';
    
    // Add rows
    data.forEach(row => {
        html += '<tr>';
        headers.forEach(header => {
            let value = row[header];
            
            // Format quality score
            if (header === 'quality_score' && value !== null) {
                value = parseFloat(value).toFixed(2);
            }
            
            // Add badge for detection type
            if (header === 'detection_type') {
                const badgeClass = value === 'HIGH-Q' ? 'badge-success' : 
                                  value === 'PERIODIC' ? 'badge-warning' : 'badge-info';
                value = `<span class="badge ${badgeClass}">${value}</span>`;
            }
            
            html += `<td>${value !== null ? value : '-'}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</tbody></table>';
    
    document.getElementById('tableContent').innerHTML = html;
}

// Show loading
function showLoading() {
    document.getElementById('tableContent').innerHTML = '<div class="loading"><div class="spinner"></div>Loading data...</div>';
}

// Show error
function showError(message) {
    document.getElementById('tableContent').innerHTML = `<div class="error">${message}</div>`;
}

// Show success
function showSuccess(message) {
    const successDiv = document.createElement('div');
    successDiv.className = 'success';
    successDiv.textContent = message;
    
    const controls = document.querySelector('.controls');
    controls.parentNode.insertBefore(successDiv, controls.nextSibling);
    
    setTimeout(() => {
        successDiv.remove();
    }, 5000);
}