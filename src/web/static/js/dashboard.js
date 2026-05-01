// ObjectIQ - Modern SaaS Dashboard JavaScript

// State
let socket;
let crowdChart = null;
let notifications = [];
let suspiciousCount = 0;
let detectionRunning = true;
let currentSource = 'webcam';

// Severity counters
let severityCounts = { low: 0, medium: 0, high: 0 };

// FPS Counter
let frameCount = 0;
let lastFpsUpdate = Date.now();

// Detection rate tracking
let previousTotalDetections = 0;
let lastRateCalculation = Date.now();

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Apply role-based visibility
    applyRoleVisibility();
    
    initializeSocket();
    initializeCrowdChart();
    loadAllData();
    startClock();
    requestNotificationPermission();
});

// Apply role-based visibility
function applyRoleVisibility() {
    if (typeof USER_ROLE !== 'undefined' && USER_ROLE === 'user') {
        document.body.classList.add('role-user');
    }
}

// Initialize Socket.IO
function initializeSocket() {
    socket = io();
    
    socket.on('connect', function() {
        updateConnectionStatus('connected', 'Connected');
    });
    
    socket.on('disconnect', function() {
        updateConnectionStatus('disconnected', 'Disconnected');
    });
    
    socket.on('new_alert', function(alert) {
        addAlert(alert);
        showBrowserNotification(alert);
    });
    
    socket.on('stats_update', function(stats) {
        updateStatistics(stats);
    });
    
    socket.on('crowd_update', function(data) {
        updateCrowdDisplay(data);
    });
    
    socket.on('alerts_history', function(alerts) {
        alerts.forEach(alert => addAlert(alert));
    });
}

// Update Connection Status
function updateConnectionStatus(status, text) {
    const dot = document.getElementById('connection-dot');
    const statusText = document.getElementById('connection-status');
    
    dot.className = 'status-dot ' + status;
    statusText.textContent = text;
}

// Initialize Crowd Chart
function initializeCrowdChart() {
    const ctx = document.getElementById('crowdChart');
    if (!ctx) return;
    
    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
    gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
    
    crowdChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'People',
                data: [],
                borderColor: '#6366f1',
                backgroundColor: gradient,
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                pointHoverRadius: 4,
                pointHoverBackgroundColor: '#6366f1',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1e293b',
                    titleColor: '#fff',
                    bodyColor: '#94a3b8',
                    borderColor: '#334155',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false
                }
            },
            scales: {
                x: {
                    display: false
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)',
                        drawBorder: false
                    },
                    ticks: {
                        color: '#64748b',
                        font: { size: 10 },
                        stepSize: 5
                    }
                }
            }
        }
    });
}

// Load All Data
async function loadAllData() {
    await Promise.all([
        loadStatistics(),
        loadCrowdData(),
        loadRecentAlerts(),
        loadSuspiciousAlerts(),
        loadRecentDetections()
    ]);
}

// Load Statistics
async function loadStatistics() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        updateStatistics(stats);
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

// Auto-refresh statistics every 5 seconds
setInterval(loadStatistics, 5000);

// Update Statistics Display
function updateStatistics(stats) {
    if (!stats) return;
    
    const totalEl = document.getElementById('stat-total');
    const avgConfEl = document.getElementById('stat-avg-conf');
    const confBar = document.getElementById('confidence-bar');
    const rateEl = document.getElementById('detection-rate');
    const rateBar = document.getElementById('rate-bar');
    
    totalEl.textContent = stats.total_detections || 0;
    
    const avgConf = (stats.average_confidence || 0) * 100;
    avgConfEl.textContent = avgConf.toFixed(1) + '%';
    confBar.style.width = avgConf + '%';
    
    // Calculate detection rate (detections per minute)
    const now = Date.now();
    const timeElapsed = (now - lastRateCalculation) / 60000; // Convert to minutes
    const currentTotal = stats.total_detections || 0;
    
    if (timeElapsed >= 1) { // Calculate every minute
        const detectionsInPeriod = currentTotal - previousTotalDetections;
        const detectionsPerMinute = Math.round(detectionsInPeriod / timeElapsed);
        
        rateEl.textContent = detectionsPerMinute + '/min';
        rateBar.style.width = Math.min(detectionsPerMinute * 2, 100) + '%';
        
        // Reset tracking
        previousTotalDetections = currentTotal;
        lastRateCalculation = now;
    }
    
    // Animate stat card
    const statCard = totalEl.closest('.stat-card');
    statCard.classList.add('detecting');
    setTimeout(() => statCard.classList.remove('detecting'), 1000);
}

// Load Crowd Data
async function loadCrowdData() {
    try {
        const response = await fetch('/api/crowd');
        const data = await response.json();
        
        document.getElementById('crowd-count').textContent = data.current_count;
        document.getElementById('live-crowd-count').textContent = data.current_count;
        
        if (crowdChart && data.history) {
            updateChart(data.history);
        }
    } catch (error) {
        console.error('Failed to load crowd data:', error);
    }
}

// Update Crowd Display
function updateCrowdDisplay(data) {
    document.getElementById('crowd-count').textContent = data.count;
    document.getElementById('live-crowd-count').textContent = data.count;
    
    if (crowdChart && data.history) {
        updateChart(data.history);
    }
}

// Update Chart
function updateChart(history) {
    const labels = history.slice(-30).map(d => {
        const date = new Date(d.timestamp);
        return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    });
    const values = history.slice(-30).map(d => d.count);
    
    crowdChart.data.labels = labels;
    crowdChart.data.datasets[0].data = values;
    crowdChart.update('none');
}

// Update Chart Range
function updateChartRange() {
    const range = document.getElementById('chart-range').value;
    loadCrowdData();
}

// Load Recent Alerts
async function loadRecentAlerts() {
    try {
        const response = await fetch('/api/alerts?limit=20');
        const alerts = await response.json();
        
        const container = document.getElementById('alerts-list');
        container.innerHTML = '';
        
        if (alerts.length === 0) {
            container.innerHTML = '<div class="no-alerts"><i class="fas fa-check-circle"></i><span>No alerts yet</span></div>';
        } else {
            alerts.forEach(alert => addAlertToList(alert));
        }
        
        updateAlertCount(alerts.length);
    } catch (error) {
        console.error('Failed to load alerts:', error);
    }
}

// Add Alert
function addAlert(alert) {
    console.log('Alert received:', alert);
    
    addAlertToList(alert);
    
    if (alert.suspicious) {
        console.log('Suspicious/Weapon alert detected:', alert.object_type);
        addSuspiciousAlert(alert);
    }
    
    addNotification(alert);
    updateAlertCount();
}

// Add Alert to List
function addAlertToList(alert) {
    const container = document.getElementById('alerts-list');
    
    // Remove "no alerts" message
    const noAlerts = container.querySelector('.no-alerts');
    if (noAlerts) noAlerts.remove();
    
    const time = alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    
    const alertHtml = `
        <div class="alert-item ${alert.suspicious ? 'suspicious' : ''}">
            <div class="alert-icon">${alert.suspicious ? '🚨' : '⚠️'}</div>
            <div class="alert-content">
                <div class="alert-title">${alert.object_type || 'Unknown'}</div>
                <div class="alert-details">${((alert.confidence || 0) * 100).toFixed(1)}% confidence</div>
            </div>
            <div class="alert-time">${time}</div>
        </div>
    `;
    
    container.insertAdjacentHTML('afterbegin', alertHtml);
    
    // Keep only 20 items
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

// Load Suspicious Alerts
async function loadSuspiciousAlerts() {
    try {
        const response = await fetch('/api/suspicious');
        const alerts = await response.json();
        
        const container = document.getElementById('suspicious-list');
        container.innerHTML = '';
        
        if (alerts.length === 0) {
            container.innerHTML = '<div class="no-alerts"><i class="fas fa-shield-alt"></i><span>No suspicious activity</span></div>';
        } else {
            alerts.forEach(alert => addSuspiciousAlert(alert));
        }
        
        suspiciousCount = alerts.length;
        document.getElementById('stat-suspicious').textContent = suspiciousCount;
        document.getElementById('suspicious-count').textContent = suspiciousCount + ' detected';
    } catch (error) {
        console.error('Failed to load suspicious alerts:', error);
    }
}

// Add Suspicious Alert
function addSuspiciousAlert(alert) {
    const container = document.getElementById('suspicious-list');
    
    const noAlerts = container.querySelector('.no-alerts');
    if (noAlerts) noAlerts.remove();
    
    suspiciousCount++;
    document.getElementById('stat-suspicious').textContent = suspiciousCount;
    document.getElementById('suspicious-count').textContent = suspiciousCount + ' detected';
    
    // Update severity counts
    const severity = alert.severity || 'medium';
    severityCounts[severity] = (severityCounts[severity] || 0) + 1;
    updateSeverityDisplay();
    
    const time = alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    
    const severityIcons = { high: '🚨', medium: '⚠️', low: '⚠' };
    const threatClass = 'threat-' + severity;
    
    const html = `
        <div class="suspicious-item ${threatClass}">
            <div class="suspicious-icon">${severityIcons[severity] || '🚨'}</div>
            <div class="suspicious-content">
                <div class="suspicious-title">${alert.object_type} <span class="severity-badge ${severity}">${severity.toUpperCase()}</span></div>
                <div class="suspicious-details">${((alert.confidence || 0) * 100).toFixed(1)}% at ${time} ${alert.person_nearby ? '• Person nearby!' : ''}</div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('afterbegin', html);
    
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

// Update Alert Count
function updateAlertCount(count) {
    const container = document.getElementById('alerts-list');
    const alertCount = count || container.querySelectorAll('.alert-item').length;
    
    document.getElementById('alert-count').textContent = alertCount + ' alerts';
    document.getElementById('nav-alert-badge').textContent = alertCount;
}

// Load Recent Detections
async function loadRecentDetections() {
    try {
        const response = await fetch('/api/detections?limit=50');
        const detections = await response.json();
        
        const tbody = document.getElementById('detections-body');
        tbody.innerHTML = '';
        
        detections.forEach(detection => addDetectionRow(detection));
    } catch (error) {
        console.error('Failed to load detections:', error);
    }
}

// Add Detection Row
function addDetectionRow(detection) {
    const tbody = document.getElementById('detections-body');
    const row = document.createElement('tr');
    
    const time = detection.timestamp ? new Date(detection.timestamp).toLocaleTimeString() : '-';
    const isSuspicious = detection.suspicious || ['gun', 'knife', 'weapon', 'stone', 'rock'].includes(detection.object_type?.toLowerCase());
    const severity = detection.severity || 'low';
    
    if (isSuspicious) row.classList.add('suspicious-row');
    
    // Update severity counts
    if (severity && severityCounts[severity] !== undefined) {
        severityCounts[severity]++;
        updateSeverityDisplay();
    }
    
    row.innerHTML = `
        <td>${time}</td>
        <td>${detection.object_type}</td>
        <td>${((detection.confidence || 0) * 100).toFixed(1)}%</td>
        <td><span class="severity-badge ${severity}">${severity}</span></td>
    `;
    
    tbody.insertBefore(row, tbody.firstChild);
    
    while (tbody.children.length > 50) {
        tbody.removeChild(tbody.lastChild);
    }
}

// Update Severity Display
function updateSeverityDisplay() {
    document.getElementById('severity-low').textContent = severityCounts.low;
    document.getElementById('severity-medium').textContent = severityCounts.medium;
    document.getElementById('severity-high').textContent = severityCounts.high;
}

// Notifications
function addNotification(alert) {
    const container = document.getElementById('notifications-list');
    const badge = document.getElementById('notification-badge');
    
    const noNotif = container.querySelector('.no-notifications');
    if (noNotif) noNotif.remove();
    
    const time = new Date().toLocaleTimeString();
    const iconClass = alert.suspicious ? 'alert' : 'warning';
    
    const html = `
        <div class="notification-item unread">
            <div class="notification-icon ${iconClass}">
                <i class="fas fa-${alert.suspicious ? 'exclamation-triangle' : 'bell'}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${alert.object_type} Detected</div>
                <div class="notification-message">Confidence: ${((alert.confidence || 0) * 100).toFixed(1)}%</div>
                <div class="notification-time">${time}</div>
            </div>
        </div>
    `;
    
    container.insertAdjacentHTML('afterbegin', html);
    
    const count = container.querySelectorAll('.notification-item').length;
    badge.textContent = count;
    badge.style.display = count > 0 ? 'block' : 'none';
}

function toggleNotifications() {
    const panel = document.getElementById('notifications-panel');
    panel.classList.toggle('open');
}

// Video Source Switching
function switchSource(source) {
    currentSource = source;
    
    // Update button states
    document.querySelectorAll('.source-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.source === source) {
            btn.classList.add('active');
        }
    });
    
    // Update indicator
    const indicator = document.getElementById('source-indicator');
    const icons = { webcam: 'webcam', video: 'film', cctv: 'camera' };
    const names = { webcam: 'Webcam', video: 'Video File', cctv: 'CCTV/RTSP' };
    indicator.innerHTML = `<i class="fas fa-${icons[source]}"></i> ${names[source]}`;
    
    // Show/hide config
    const config = document.getElementById('source-config');
    const videoConfig = document.getElementById('video-config');
    const cctvConfig = document.getElementById('cctv-config');
    
    config.style.display = source !== 'webcam' ? 'block' : 'none';
    videoConfig.style.display = source === 'video' ? 'flex' : 'none';
    cctvConfig.style.display = source === 'cctv' ? 'flex' : 'none';
    
    // Send API request
    if (source === 'webcam') {
        fetch('/api/source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: 'webcam' })
        });
    }
}

function loadVideoFile(input) {
    if (input.files && input.files[0]) {
        const formData = new FormData();
        formData.append('video', input.files[0]);
        
        // Show loading indicator
        const indicator = document.getElementById('source-indicator');
        indicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
        
        fetch('/api/source/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log('Video uploaded:', data);
            if (data.status === 'ok') {
                indicator.innerHTML = '<i class="fas fa-film"></i> ' + data.filename;
                // Update source buttons
                document.querySelectorAll('.source-btn').forEach(btn => {
                    btn.classList.remove('active');
                    if (btn.dataset.source === 'video') {
                        btn.classList.add('active');
                    }
                });
            } else {
                indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Upload failed';
                setTimeout(() => {
                    indicator.innerHTML = '<i class="fas fa-webcam"></i> Webcam';
                }, 2000);
            }
        })
        .catch(err => {
            console.error('Upload error:', err);
            indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Upload error';
        });
    }
}

function connectRTSP() {
    const url = document.getElementById('rtsp-url').value;
    if (url) {
        const indicator = document.getElementById('source-indicator');
        indicator.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Connecting...';
        
        fetch('/api/source', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source: 'rtsp', url: url })
        })
        .then(response => response.json())
        .then(data => {
            console.log('RTSP connected:', data);
            if (data.status === 'ok') {
                indicator.innerHTML = '<i class="fas fa-camera"></i> CCTV Connected';
            } else {
                indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Connection failed';
            }
        })
        .catch(err => {
            console.error('RTSP error:', err);
            indicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Connection error';
        });
    }
}

// Toggle Detection
function toggleDetection() {
    detectionRunning = !detectionRunning;
    const btn = document.getElementById('detection-toggle');
    btn.innerHTML = detectionRunning ? '<i class="fas fa-pause"></i>' : '<i class="fas fa-play"></i>';
    
    fetch('/api/detection/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ running: detectionRunning })
    });
}

// Fullscreen
function toggleFullscreen() {
    const container = document.getElementById('video-container');
    
    if (document.fullscreenElement) {
        document.exitFullscreen();
    } else {
        container.requestFullscreen();
    }
}

// Clear Functions
function clearDetections() {
    document.getElementById('detections-body').innerHTML = '';
}

function clearAllAlerts() {
    document.getElementById('alerts-list').innerHTML = '<div class="no-alerts"><i class="fas fa-check-circle"></i><span>No alerts yet</span></div>';
    document.getElementById('suspicious-list').innerHTML = '<div class="no-alerts"><i class="fas fa-shield-alt"></i><span>No suspicious activity</span></div>';
    document.getElementById('alert-count').textContent = '0 alerts';
    document.getElementById('suspicious-count').textContent = '0 detected';
    document.getElementById('stat-suspicious').textContent = '0';
    suspiciousCount = 0;
}

// Export Data
function exportData(format) {
    if (format === 'csv') {
        window.open('/api/report/export-csv', '_blank');
    } else if (format === 'pdf') {
        window.open('/api/report/generate', '_blank');
    }
}

// Sidebar Toggle
function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    sidebar.classList.toggle('open');
}

// Clock
function startClock() {
    function updateClock() {
        const now = new Date();
        const options = { 
            weekday: 'short', 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        };
        document.getElementById('current-time').textContent = now.toLocaleDateString('en-US', options);
        
        // Overlay time
        const overlayTime = document.getElementById('overlay-time');
        if (overlayTime) {
            overlayTime.innerHTML = `<i class="fas fa-clock"></i> ${now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}`;
        }
    }
    
    updateClock();
    setInterval(updateClock, 1000);
}

// FPS Counter (called from video frame updates)
function updateFps() {
    frameCount++;
    const now = Date.now();
    
    if (now - lastFpsUpdate >= 1000) {
        document.getElementById('fps').textContent = frameCount;
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

// Browser Notification
function requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission();
    }
}

function showBrowserNotification(alert) {
    if (!('Notification' in window) || Notification.permission !== 'granted') return;
    
    const title = alert.suspicious ? 
        `🚨 SUSPICIOUS: ${alert.object_type}` : 
        `${alert.object_type} Detected`;
    
    new Notification(title, {
        body: `Confidence: ${((alert.confidence || 0) * 100).toFixed(1)}%`,
        icon: '/static/img/icon.png'
    });
}

// Simulate FPS updates (in real app, this would come from server)
setInterval(updateFps, 100);

// ==================== Section Navigation ====================
function showSection(section) {
    // Hide all sections
    document.querySelectorAll('.dashboard-content, .analytics-section, .alerts-section, .settings-section, .reports-section').forEach(el => {
        el.style.display = 'none';
    });
    
    // Show selected section
    if (section === 'dashboard') {
        document.querySelector('.dashboard-content').style.display = 'block';
    } else {
        const sectionEl = document.getElementById(section + '-section');
        if (sectionEl) {
            sectionEl.style.display = 'block';
            
            // Load section-specific data
            if (section === 'analytics') {
                loadAnalytics();
            } else if (section === 'alerts') {
                loadAlertsSection();
            } else if (section === 'settings') {
                loadSettings();
            } else if (section === 'reports') {
                loadReportsSection();
            }
        }
    }
    
    // Update nav active state
    document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
    event.target.closest('.nav-item').classList.add('active');
}

// ==================== Reports Section ====================
function loadReportsSection() {
    // Set default dates
    const today = new Date();
    const weekAgo = new Date(today.getTime() - 7 * 24 * 60 * 60 * 1000);
    
    document.getElementById('report-end-date').value = today.toISOString().split('T')[0];
    document.getElementById('report-start-date').value = weekAgo.toISOString().split('T')[0];
    
    // Load stats
    loadReportStats();
}

async function loadReportStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        document.getElementById('report-total').textContent = stats.total_detections || 0;
        document.getElementById('report-alerts').textContent = stats.alerts_today || 0;
        document.getElementById('report-suspicious').textContent = stats.suspicious_count || 0;
        document.getElementById('report-crowd').textContent = stats.max_crowd || 0;
    } catch (error) {
        console.error('Failed to load report stats:', error);
    }
}

function generatePDF() {
    const start = document.getElementById('report-start-date').value;
    const end = document.getElementById('report-end-date').value;
    if (!start || !end) {
        alert('Please select both start and end dates');
        return;
    }
    window.open(`/api/report/generate?start=${start}&end=${end}`, '_blank');
}

function exportCSV() {
    const start = document.getElementById('report-start-date').value;
    const end = document.getElementById('report-end-date').value;
    if (!start || !end) {
        alert('Please select both start and end dates');
        return;
    }
    window.open(`/api/report/export-csv?start=${start}&end=${end}`, '_blank');
}

// ==================== Analytics Section ==================
let detectionChart = null;
let threatChart = null;
let hourlyChart = null;
let weeklyChart = null;

async function loadAnalytics() {
    try {
        const response = await fetch('/api/analytics');
        const data = await response.json();
        
        console.log('Analytics data received:', data);
        console.log('Threat analysis data:', data.threat_analysis);
        
        updateAnalyticsCharts(data);
        updateThreatStats(data.threat_analysis);
        loadHeatmap();
    } catch (error) {
        console.error('Failed to load analytics:', error);
    }
}

function updateAnalyticsCharts(data) {
    // Detection Distribution Chart
    if (!detectionChart) {
        const ctx = document.getElementById('detection-chart');
        if (ctx) {
            detectionChart = new Chart(ctx.getContext('2d'), {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: ['#6366f1', '#22c55e', '#f59e0b', '#ef4444', '#8b5cf6']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }
    }
    
    // Update with object counts
    if (data.detection_stats && data.detection_stats.object_counts) {
        const counts = data.detection_stats.object_counts;
        detectionChart.data.labels = Object.keys(counts);
        detectionChart.data.datasets[0].data = Object.values(counts);
        detectionChart.update();
    }
    
    // Hourly Activity Chart
    if (!hourlyChart) {
        const ctx = document.getElementById('hourly-chart');
        if (ctx) {
            hourlyChart = new Chart(ctx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Detections',
                        data: [],
                        backgroundColor: '#6366f1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        x: { grid: { display: false } },
                        y: { beginAtZero: true }
                    }
                }
            });
        }
    }
    
    if (data.hourly_activity) {
        hourlyChart.data.labels = data.hourly_activity.map(h => h.hour);
        hourlyChart.data.datasets[0].data = data.hourly_activity.map(h => h.count);
        hourlyChart.update();
    }
    
    // Weekly Trends Chart
    if (!weeklyChart) {
        const ctx = document.getElementById('weekly-chart');
        if (ctx) {
            const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
            gradient.addColorStop(0, 'rgba(99, 102, 241, 0.3)');
            gradient.addColorStop(1, 'rgba(99, 102, 241, 0.0)');
            
            weeklyChart = new Chart(ctx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Detections',
                        data: [],
                        borderColor: '#6366f1',
                        backgroundColor: gradient,
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#6366f1',
                        pointBorderColor: '#fff',
                        pointBorderWidth: 2,
                        borderWidth: 3
                    }]
                },
                options: {
                    responsive: true,
                    interaction: {
                        intersect: false,
                        mode: 'index'
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: '#1e293b',
                            titleColor: '#fff',
                            bodyColor: '#94a3b8',
                            borderColor: '#334155',
                            borderWidth: 1,
                            padding: 12
                        }
                    },
                    scales: {
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#64748b',
                                font: { size: 11 }
                            }
                        },
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.05)',
                                drawBorder: false
                            },
                            ticks: {
                                color: '#64748b',
                                font: { size: 11 },
                                stepSize: 5
                            }
                        }
                    }
                }
            });
        }
    }
    
    // Update weekly chart with data or generate from hourly data
    if (data.weekly_trends && data.weekly_trends.length > 0) {
        weeklyChart.data.labels = data.weekly_trends.map(w => w.day + ' (' + w.date + ')');
        weeklyChart.data.datasets[0].data = data.weekly_trends.map(w => w.count);
        weeklyChart.update();
    } else if (data.hourly_activity) {
        // Generate weekly-like data from hourly activity (last 7 hours as fallback)
        const last7Hours = data.hourly_activity.slice(-7);
        weeklyChart.data.labels = last7Hours.map(h => h.hour);
        weeklyChart.data.datasets[0].data = last7Hours.map(h => h.count);
        weeklyChart.update();
    }
}

function updateThreatStats(threatData) {
    console.log('Threat Data Received:', threatData);
    
    if (!threatData) {
        console.warn('No threat data available');
        return;
    }
    
    const totalEl = document.getElementById('threat-total');
    const highEl = document.getElementById('threat-high');
    const mediumEl = document.getElementById('threat-medium');
    const lowEl = document.getElementById('threat-low');
    
    if (totalEl) totalEl.textContent = threatData.total || 0;
    if (highEl) highEl.textContent = threatData.high || 0;
    if (mediumEl) mediumEl.textContent = threatData.medium || 0;
    if (lowEl) lowEl.textContent = threatData.low || 0;
    
    console.log('Threat stats updated:', {
        total: threatData.total || 0,
        high: threatData.high || 0,
        medium: threatData.medium || 0,
        low: threatData.low || 0
    });
    
    // Threat Chart
    if (!threatChart) {
        const ctx = document.getElementById('threat-chart');
        if (ctx) {
            threatChart = new Chart(ctx.getContext('2d'), {
                type: 'bar',
                data: {
                    labels: ['High', 'Medium', 'Low'],
                    datasets: [{
                        label: 'Threats',
                        data: [threatData.high || 0, threatData.medium || 0, threatData.low || 0],
                        backgroundColor: ['#ef4444', '#f59e0b', '#22c55e']
                    }]
                },
                options: {
                    responsive: true,
                    indexAxis: 'y',
                    plugins: { legend: { display: false } }
                }
            });
        }
    } else {
        threatChart.data.datasets[0].data = [threatData.high || 0, threatData.medium || 0, threatData.low || 0];
        threatChart.update();
    }
}

async function loadHeatmap() {
    try {
        const response = await fetch('/api/heatmap');
        if (response.ok) {
            const data = await response.json();
            // Update heatmap image if available
            const heatmapImg = document.getElementById('heatmap-image');
            if (heatmapImg && data.image) {
                heatmapImg.src = 'data:image/png;base64,' + data.image;
            }
        }
    } catch (error) {
        console.error('Failed to load heatmap:', error);
    }
}

function refreshHeatmap() {
    loadHeatmap();
}

// ==================== Alerts Section ====================
async function loadAlertsSection() {
    try {
        const response = await fetch('/api/alerts?limit=100');
        const alerts = await response.json();
        
        const container = document.getElementById('alerts-list-container');
        container.innerHTML = '';
        
        alerts.forEach(alert => {
            const alertHtml = createAlertCard(alert);
            container.insertAdjacentHTML('beforeend', alertHtml);
        });
    } catch (error) {
        console.error('Failed to load alerts:', error);
    }
}

function createAlertCard(alert) {
    const time = alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'N/A';
    const severity = alert.severity || 'low';
    const isSuspicious = alert.suspicious || false;
    
    return `
        <div class="alert-card ${isSuspicious ? 'suspicious' : ''} ${severity}" onclick="showAlertDetail('${alert._id || alert.timestamp}')">
            <div class="alert-card-icon">
                ${isSuspicious ? '<i class="fas fa-exclamation-triangle"></i>' : '<i class="fas fa-bell"></i>'}
            </div>
            <div class="alert-card-content">
                <div class="alert-card-title">${alert.object_type || 'Unknown'}</div>
                <div class="alert-card-meta">
                    <span>${((alert.confidence || 0) * 100).toFixed(1)}% confidence</span>
                    <span class="severity-badge ${severity}">${severity}</span>
                </div>
                <div class="alert-card-time">${time}</div>
            </div>
        </div>
    `;
}

function showAlertDetail(alertId) {
    // Load and display alert details
    const detailContent = document.getElementById('alert-detail-content');
    // In a real implementation, this would fetch the alert details
    detailContent.innerHTML = '<p>Loading...</p>';
}

// ==================== Settings Section ====================
function showSettingsTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.settings-tab').forEach(t => t.style.display = 'none');
    
    // Show selected tab
    const tabEl = document.getElementById(tab + '-settings');
    if (tabEl) {
        tabEl.style.display = 'block';
    }
    
    // Update nav active state
    document.querySelectorAll('.settings-tab-btn').forEach(item => item.classList.remove('active'));
    event.target.closest('.settings-tab-btn').classList.add('active');
}

async function loadSettings() {
    try {
        const response = await fetch('/api/settings');
        const settings = await response.json();
        
        // Detection settings
        const confSlider = document.getElementById('confidence-threshold');
        if (confSlider) {
            confSlider.value = settings.detection?.confidence_threshold || 0.5;
            document.getElementById('confidence-value').textContent = 
                ((settings.detection?.confidence_threshold || 0.5) * 100).toFixed(0) + '%';
        }
        
        // Load users if admin
        if (IS_ADMIN) {
            loadUsers();
        }
    } catch (error) {
        console.error('Failed to load settings:', error);
    }
}

async function loadUsers() {
    try {
        const response = await fetch('/api/users');
        const users = await response.json();
        
        const tbody = document.getElementById('users-table-body');
        tbody.innerHTML = '';
        
        users.forEach(user => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>
                    <div class="user-cell">
                        <div class="user-avatar-small">
                            ${user.profile_image_url ? 
                                '<img src="' + user.profile_image_url + '">' : 
                                '<i class="fas fa-user"></i>'}
                        </div>
                        <span>${user.username}</span>
                    </div>
                </td>
                <td>${user.email}</td>
                <td><span class="role-badge ${user.role}">${user.role}</span></td>
                <td><span class="status-badge ${user.is_active ? 'active' : 'inactive'}">${user.is_active ? 'Active' : 'Inactive'}</span></td>
                <td>${user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}</td>
                <td>
                    <button onclick="editUser('${user.id}')" class="btn-icon"><i class="fas fa-edit"></i></button>
                    <button onclick="deleteUser('${user.id}')" class="btn-icon danger"><i class="fas fa-trash"></i></button>
                </td>
            `;
            tbody.appendChild(row);
        });
    } catch (error) {
        console.error('Failed to load users:', error);
    }
}

// ==================== Export Modal ====================
function toggleExportModal() {
    const modal = document.getElementById('export-modal');
    modal.classList.toggle('open');
}

// Export type change handler
document.querySelectorAll('input[name="export-type"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const emailInput = document.getElementById('export-email-input');
        emailInput.style.display = this.value === 'email' ? 'block' : 'none';
    });
});

async function confirmExport() {
    const exportType = document.querySelector('input[name="export-type"]:checked').value;
    const startDate = document.getElementById('export-start-date').value;
    const endDate = document.getElementById('export-end-date').value;
    
    if (exportType === 'email') {
        const email = document.getElementById('export-recipient-email').value;
        if (!email) {
            showNotification('Please enter an email address', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/export/email', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    email: email,
                    start_date: startDate,
                    end_date: endDate
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'ok') {
                showNotification('Export sent to ' + email, 'success');
                toggleExportModal();
            } else {
                showNotification(result.message || 'Export failed', 'error');
            }
        } catch (error) {
            console.error('Export error:', error);
            showNotification('Export failed', 'error');
        }
    } else if (exportType === 'csv') {
        window.open('/api/report/export-csv', '_blank');
        toggleExportModal();
    } else if (exportType === 'pdf') {
        window.open('/api/report/generate', '_blank');
        toggleExportModal();
    }
}

// ==================== Settings Forms ====================
// Detection settings form
document.getElementById('detection-settings-form')?.addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const data = {
        confidence_threshold: parseFloat(document.getElementById('confidence-threshold').value),
        frame_skip: parseInt(document.getElementById('frame-skip').value),
        target_objects: Array.from(document.querySelectorAll('input[name="target-objects"]:checked')).map(cb => cb.value),
        crowd_threshold: parseInt(document.getElementById('crowd-threshold').value)
    };
    
    try {
        const response = await fetch('/api/settings/detection', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('Detection settings saved!', 'success');
        } else {
            showNotification(result.message || 'Failed to save settings', 'error');
        }
    } catch (error) {
        console.error('Settings save error:', error);
        showNotification('Failed to save settings', 'error');
    }
});

// Confidence slider update
document.getElementById('confidence-threshold')?.addEventListener('input', function() {
    document.getElementById('confidence-value').textContent = (this.value * 100).toFixed(0) + '%';
});

// Weapon sensitivity slider
document.getElementById('weapon-sensitivity')?.addEventListener('input', function() {
    document.getElementById('weapon-sensitivity-value').textContent = (this.value * 100).toFixed(0) + '%';
});

// Video source change handler
document.getElementById('video-source-select')?.addEventListener('change', function() {
    const source = this.value;
    document.getElementById('rtsp-settings').style.display = source === 'rtsp' ? 'block' : 'none';
    document.getElementById('video-file-settings').style.display = source === 'video' ? 'block' : 'none';
});

// ==================== User Management ====================
function showAddUserModal() {
    // Show modal for adding new user
    const username = prompt('Enter username:');
    const email = prompt('Enter email:');
    const password = prompt('Enter password:');
    const role = prompt('Enter role (admin/user):', 'user');
    
    if (username && email && password) {
        createUser(username, email, password, role);
    }
}

async function createUser(username, email, password, role) {
    try {
        const response = await fetch('/api/users', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password, role })
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('User created successfully!', 'success');
            loadUsers();
        } else {
            showNotification(result.message || 'Failed to create user', 'error');
        }
    } catch (error) {
        console.error('Create user error:', error);
        showNotification('Failed to create user', 'error');
    }
}

async function deleteUser(userId) {
    if (!confirm('Are you sure you want to delete this user?')) return;
    
    try {
        const response = await fetch('/api/users/' + userId, {
            method: 'DELETE'
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification('User deleted!', 'success');
            loadUsers();
        } else {
            showNotification(result.message || 'Failed to delete user', 'error');
        }
    } catch (error) {
        console.error('Delete user error:', error);
        showNotification('Failed to delete user', 'error');
    }
}

function editUser(userId) {
    // Show edit modal
    alert('Edit user functionality - would show modal with user details');
}

// ==================== System Actions ====================
async function clearOldRecords() {
    if (!confirm('This will delete old detection records. Continue?')) return;
    
    try {
        const response = await fetch('/api/system/cleanup', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (result.status === 'ok') {
            showNotification(`Cleaned up ${result.deleted} old records`, 'success');
        } else {
            showNotification(result.message || 'Cleanup failed', 'error');
        }
    } catch (error) {
        console.error('Cleanup error:', error);
        showNotification('Cleanup failed', 'error');
    }
}

function exportDatabase() {
    window.open('/api/report/export-csv', '_blank');
}

function restartSystem() {
    if (!confirm('Are you sure you want to restart the system?')) return;
    // Would call API to restart system
    showNotification('System restart initiated...', 'info');
}

// ==================== Notifications ====================
function showNotification(message, type = 'info') {
    const container = document.createElement('div');
    container.className = `toast-notification ${type}`;
    container.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(container);
    
    // Animate in
    setTimeout(() => container.classList.add('show'), 10);
    
    // Remove after 3 seconds
    setTimeout(() => {
        container.classList.remove('show');
        setTimeout(() => container.remove(), 300);
    }, 3000);
}

// ==================== ROI Drawing ====================
let isDrawingROI = false;
let roiPoints = [];

function startROIDrawing() {
    isDrawingROI = true;
    showNotification('Click on video to set ROI points. Double-click to finish.', 'info');
}

// ==================== Utility Functions ====================
// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// Format date
function formatDate(date) {
    return new Date(date).toLocaleString();
}
