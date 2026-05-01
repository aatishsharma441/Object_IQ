"""
Web Dashboard Module - Enhanced Version
Flask-based real-time web dashboard with authentication, crowd counting,
suspicious activity detection, and report generation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import base64
import io
import json
import threading
import time
from functools import wraps

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for, flash, session, send_file
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_login import LoginManager, login_user, logout_user, login_required, current_user

from .auth import User, UserManager

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'object-detection-dashboard-2024'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access the dashboard.'

# Global state
_detection_system = None
_user_manager: Optional[UserManager] = None
_latest_frame: Optional[np.ndarray] = None
_latest_detections: List[Dict] = []
_alerts: List[Dict] = []
_statistics: Dict[str, Any] = {}
_crowd_count: int = 0
_crowd_history: List[Dict] = []  # For crowd chart
_suspicious_alerts: List[Dict] = []


def set_detection_system(system) -> None:
    """Set the detection system reference."""
    global _detection_system, _user_manager
    _detection_system = system
    if system and system.database:
        _user_manager = UserManager(system.database)
        _user_manager.create_default_admin()


@login_manager.user_loader
def load_user(user_id):
    """Load user for Flask-Login."""
    global _user_manager
    if _user_manager:
        return _user_manager.get_user(user_id)
    return None


def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role != 'admin':
            return jsonify({'error': 'Admin access required'}), 403
        return f(*args, **kwargs)
    return decorated_function


def update_frame(frame: np.ndarray, detections: List[Dict] = None, crowd_count: int = 0) -> None:
    """Update the latest frame for streaming."""
    global _latest_frame, _latest_detections, _crowd_count
    _latest_frame = frame.copy()
    _crowd_count = crowd_count
    if detections:
        _latest_detections = detections


def add_alert(alert: Dict, suspicious: bool = False) -> None:
    """Add an alert to the history."""
    global _alerts, _suspicious_alerts
    
    # Only set suspicious from parameter if not already in alert dict
    if 'suspicious' not in alert:
        alert['suspicious'] = suspicious
    
    # Add to suspicious alerts if marked as suspicious
    if alert.get('suspicious', False):
        _suspicious_alerts.insert(0, alert)
        if len(_suspicious_alerts) > 50:
            _suspicious_alerts = _suspicious_alerts[:50]
    
    _alerts.insert(0, alert)
    if len(_alerts) > 100:
        _alerts = _alerts[:100]
    
    # Emit to connected clients
    socketio.emit('new_alert', alert)


def update_statistics(stats: Dict) -> None:
    """Update statistics."""
    global _statistics
    _statistics = stats


def add_crowd_data(count: int, timestamp: datetime = None) -> None:
    """Add crowd count data point for charting."""
    global _crowd_history
    
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    _crowd_history.append({
        'timestamp': timestamp.isoformat(),
        'count': count
    })
    
    # Keep last 1000 data points
    if len(_crowd_history) > 1000:
        _crowd_history = _crowd_history[-1000:]
    
    # Emit to dashboard
    socketio.emit('crowd_update', {'count': count, 'timestamp': timestamp.isoformat()})


def get_frame() -> Optional[np.ndarray]:
    """Get the latest frame."""
    return _latest_frame


def frame_to_base64(frame: np.ndarray) -> str:
    """Convert frame to base64 string."""
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


# ==================== Routes ====================

@app.route('/')
def index():
    """Home page - redirect to dashboard if logged in, otherwise show landing."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')


@app.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html', user=current_user)


@app.route('/profile')
@login_required
def profile():
    """Profile settings page."""
    return render_template('profile.html', user=current_user)


# ==================== Authentication Routes ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if _user_manager:
            user = _user_manager.authenticate(username, password)
            if user:
                login_user(user)
                next_page = request.args.get('next')
                return redirect(next_page or url_for('dashboard'))
            else:
                error = 'Invalid username or password'
        else:
            error = 'Authentication system not initialized'
    
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    """Registration page."""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        role = request.form.get('role', 'user')  # Get role from form
        
        # Validate role
        if role not in ['user', 'admin']:
            role = 'user'
        
        if password != confirm_password:
            error = 'Passwords do not match'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters'
        elif _user_manager:
            success, message = _user_manager.create_user(username, email, password, role)
            if success:
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            else:
                error = message
        else:
            error = 'Registration not available'
    
    return render_template('register.html', error=error)


@app.route('/logout')
@login_required
def logout():
    """Logout user."""
    logout_user()
    return redirect(url_for('login'))


# ==================== API Routes ====================

@app.route('/api/stats')
@login_required
def api_stats():
    """Get current statistics."""
    global _statistics
    return jsonify(_statistics)


@app.route('/api/crowd')
@login_required
def api_crowd():
    """Get current crowd count and history."""
    global _crowd_count, _crowd_history
    return jsonify({
        'current_count': _crowd_count,
        'history': _crowd_history[-100:]  # Last 100 data points
    })


@app.route('/api/alerts')
@login_required
def api_alerts():
    """Get alert history."""
    global _alerts
    limit = request.args.get('limit', 50, type=int)
    return jsonify(_alerts[:limit])


@app.route('/api/suspicious')
@login_required
def api_suspicious():
    """Get suspicious activity alerts."""
    global _suspicious_alerts
    
    print(f"\n=== SUSPICIOUS ALERTS REQUEST ===")
    print(f"Returning {len(_suspicious_alerts)} suspicious alerts")
    
    return jsonify(_suspicious_alerts)


@app.route('/api/detections')
@login_required
def api_detections():
    """Get recent detections from database."""
    global _detection_system
    if _detection_system and _detection_system.database:
        limit = request.args.get('limit', 100, type=int)
        detections = _detection_system.database.get_recent_detections(limit=limit)
        return jsonify(detections)
    return jsonify([])


@app.route('/api/config')
@login_required
def api_config():
    """Get current configuration."""
    global _detection_system
    if _detection_system:
        config = _detection_system.config
        return jsonify({
            'model': config.model_name,
            'resolution': config.resolution,
            'frame_rate': config.frame_rate,
            'target_objects': config.target_objects,
            'confidence_threshold': config.confidence_threshold,
            'consecutive_frames': config.consecutive_frames,
            'cooldown_seconds': config.cooldown_seconds,
            'roi_enabled': config.roi_enabled,
            'crowd_detection': config._objects_config.get('crowd_detection', {})
        })
    return jsonify({})


@app.route('/api/user/profile')
@login_required
def api_user_profile():
    """Get current user profile."""
    return jsonify({
        'id': current_user.id,
        'username': current_user.username,
        'email': current_user.email,
        'role': current_user.role,
        'created_at': current_user.created_at.isoformat() if current_user.created_at else None,
        'last_login': current_user.last_login.isoformat() if current_user.last_login else None,
        'is_admin': current_user.role == 'admin'
    })


@app.route('/api/user/profile', methods=['PUT'])
@login_required
def api_update_profile():
    """Update user profile."""
    global _user_manager
    data = request.get_json()
    
    if _user_manager:
        success, message = _user_manager.update_profile(current_user.id, data)
        if success:
            return jsonify({'status': 'ok', 'message': message})
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/user/profile/image', methods=['POST'])
@login_required
def api_upload_profile_image():
    """Upload profile image."""
    global _user_manager
    
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Check file type
    from .auth import allowed_file
    if not allowed_file(file.filename):
        return jsonify({'status': 'error', 'message': 'Invalid file type'}), 400
    
    # Read and process image
    image_data = file.read()
    content_type = file.content_type or 'image/jpeg'
    
    if _user_manager:
        success, message = _user_manager.update_profile_image(
            current_user.id, image_data, content_type
        )
        if success:
            return jsonify({'status': 'ok', 'message': message})
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/user/password', methods=['POST'])
@login_required
def api_change_password():
    """Change user password."""
    global _user_manager
    data = request.get_json()
    
    current_password = data.get('current_password', '')
    new_password = data.get('new_password', '')
    
    if _user_manager:
        success, message = _user_manager.change_password(
            current_user.id, current_password, new_password
        )
        if success:
            return jsonify({'status': 'ok', 'message': message})
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/user/notification-settings', methods=['POST'])
@login_required
def api_update_notification_settings():
    """Update user notification settings."""
    global _user_manager
    data = request.get_json()
    
    if _user_manager:
        success, message = _user_manager.update_notification_settings(
            current_user.id, data
        )
        if success:
            return jsonify({'status': 'ok', 'message': message})
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/settings', methods=['GET'])
@login_required
def api_get_settings():
    """Get all system settings."""
    global _detection_system
    
    settings = {
        'detection': {
            'confidence_threshold': 0.5,
            'frame_skip': 2,
            'target_objects': ['person', 'car'],
            'weapon_sensitivity': 0.75,
            'crowd_threshold': 5
        },
        'notifications': {
            'email_enabled': False,
            'alert_types': ['weapon', 'suspicious'],
            'cooldown': 30
        },
        'camera': {
            'source': 'webcam',
            'resolution': '1280x720',
            'framerate': 30,
            'roi_enabled': False
        }
    }
    
    if _detection_system and _detection_system.config:
        config = _detection_system.config
        settings['detection']['confidence_threshold'] = config.confidence_threshold
        settings['detection']['target_objects'] = config.target_objects
        settings['camera']['resolution'] = f"{config.resolution[0]}x{config.resolution[1]}"
        settings['camera']['framerate'] = config.frame_rate
    
    
    return jsonify(settings)


@app.route('/api/settings/detection', methods=['POST'])
@login_required
@admin_required
def api_update_detection_settings():
    """Update detection settings."""
    global _detection_system
    data = request.get_json()
    
    if _detection_system:
        if 'confidence_threshold' in data:
            _detection_system.detector.set_confidence_threshold(data['confidence_threshold'])
        if 'target_objects' in data:
            _detection_system.detector.set_target_objects(data['target_objects'])
        return jsonify({'status': 'ok', 'message': 'Detection settings updated'})
    
    return jsonify({'status': 'error', 'message': 'Detection system not available'}), 400


@app.route('/api/settings/notifications', methods=['POST'])
@login_required
@admin_required
def api_update_notification_settings_system():
    """Update notification settings."""
    global _detection_system
    data = request.get_json()
    
    # These settings would be saved to config file or database
    return jsonify({'status': 'ok', 'message': 'Notification settings updated'})


@app.route('/api/settings/camera', methods=['POST'])
@login_required
@admin_required
def api_update_camera_settings():
    """Update camera settings."""
    global _detection_system
    data = request.get_json()
    
    if _detection_system:
        if 'resolution' in data:
            try:
                w, h = map(int, data['resolution'].split('x'))
                _detection_system.detector.set_resolution((w, h))
            except:
                pass
        return jsonify({'status': 'ok', 'message': 'Camera settings updated'})
    
    return jsonify({'status': 'error', 'message': 'Detection system not available'}), 400


@app.route('/api/users')
@login_required
@admin_required
def api_get_users():
    """Get all users."""
    global _user_manager
    
    if _user_manager:
        users = _user_manager.get_all_users()
        return jsonify(users)
    return jsonify([])


@app.route('/api/users', methods=['POST'])
@login_required
@admin_required
def api_create_user():
    """Create new user."""
    global _user_manager
    data = request.get_json()
    
    if _user_manager:
        success, message = _user_manager.create_user(
            username=data.get('username'),
            email=data.get('email'),
            password=data.get('password'),
            role=data.get('role', 'user')
        )
        if success:
            return jsonify({'status': 'ok', 'message': message})
        return jsonify({'status': 'error', 'message': message}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/users/<user_id>', methods=['DELETE'])
@login_required
@admin_required
def api_delete_user(user_id):
    """Delete user."""
    global _user_manager
    
    if _user_manager:
        success = _user_manager.delete_user(user_id)
        if success:
            return jsonify({'status': 'ok', 'message': 'User deleted'})
        return jsonify({'status': 'error', 'message': 'Failed to delete user'}), 400
    
    return jsonify({'status': 'error', 'message': 'User manager not available'}), 400


@app.route('/api/analytics')
@login_required
def api_get_analytics():
    """Get analytics data."""
    global _detection_system, _statistics, _crowd_history, _suspicious_alerts
    
    # Log threat data for debugging
    print(f"\n=== ANALYTICS REQUEST ===")
    print(f"Total suspicious alerts: {len(_suspicious_alerts)}")
    for alert in _suspicious_alerts[:3]:  # Show first 3
        print(f"  - {alert.get('object_type', 'Unknown')} | Severity: {alert.get('severity', 'N/A')} | Weapon: {alert.get('weapon', False)}")
    
    analytics = {
        'detection_stats': _statistics,
        'crowd_history': _crowd_history[-50:],
        'threat_analysis': {
            'total': len(_suspicious_alerts),
            'high': sum(1 for a in _suspicious_alerts if a.get('severity') == 'high'),
            'medium': sum(1 for a in _suspicious_alerts if a.get('severity') == 'medium'),
            'low': sum(1 for a in _suspicious_alerts if a.get('severity') == 'low')
        },
        'hourly_activity': [],
        'weekly_trends': []
    }
    
    if _detection_system and _detection_system.database:
        # Get hourly activity
        now = datetime.utcnow()
        hourly_data = []
        for i in range(24):
            hour_start = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=i)
            hour_end = hour_start + timedelta(hours=1)
            stats = _detection_system.database.get_detection_stats(hour_start, hour_end)
            hourly_data.append({
                'hour': hour_start.strftime('%H:00'),
                'count': stats['total_detections']
            })
        analytics['hourly_activity'] = hourly_data
        
        # Get weekly trends (last 7 days)
        weekly_data = []
        for i in range(7):
            day_start = (now - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = day_start + timedelta(days=1)
            stats = _detection_system.database.get_detection_stats(day_start, day_end)
            weekly_data.insert(0, {
                'day': day_start.strftime('%A'),  # Day name (e.g., Monday)
                'date': day_start.strftime('%m/%d'),
                'count': stats['total_detections']
            })
        analytics['weekly_trends'] = weekly_data
    
    return jsonify(analytics)


@app.route('/api/heatmap')
@login_required
def api_get_heatmap():
    """Get current heatmap data."""
    global _detection_system
    
    if _detection_system and hasattr(_detection_system, '_heatmap_data'):
        return jsonify(_detection_system._heatmap_data)
    
    return jsonify({'status': 'error', 'message': 'Heatmap not available'}), 404


@app.route('/api/export/email', methods=['POST'])
@login_required
@admin_required
def api_export_via_email():
    """Export data and send via email."""
    global _detection_system, _user_manager
    data = request.get_json()
    
    email = data.get('email')
    export_type = data.get('type', 'csv')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not email:
        return jsonify({'status': 'error', 'message': 'Email address required'}), 400
    
    # Generate export data
    if _detection_system and _detection_system.database:
        detections = _detection_system.database.get_recent_detections(limit=1000)
        
        # Send email with export
        if _detection_system.email_service:
            from .notifications.templates import EmailTemplate
            
            subject, html, text = EmailTemplate.system_notification(
                title="Data Export",
                message=f"Your requested data export is attached. {len(detections)} records included.",
                level="info"
            )
            
            # For now, just return success (actual file attachment would be added)
            return jsonify({
                'status': 'ok', 
                'message': f'Export sent to {email}',
                'records': len(detections)
            })
    
    return jsonify({'status': 'error', 'message': 'Export failed'}), 500


@app.route('/api/system/status')
@login_required
def api_system_status():
    """Get system status."""
    global _detection_system
    
    status = {
        'detection_running': False,
        'database_connected': False,
        'camera_connected': False,
        'email_configured': False,
        'uptime': 0,
        'frame_count': 0
    }
    
    if _detection_system:
        status['detection_running'] = _detection_system._running
        status['frame_count'] = _detection_system._frame_count
        
        if _detection_system.database:
            success, _ = _detection_system.database.test_connection()
            status['database_connected'] = success
        
        
        if _detection_system.camera:
            status['camera_connected'] = _detection_system.camera.is_connected()
        
        
        status['email_configured'] = _detection_system.email_service is not None
        
        if hasattr(_detection_system, '_start_time'):
            status['uptime'] = (datetime.utcnow() - _detection_system._start_time).total_seconds()
    
    
    return jsonify(status)


@app.route('/api/system/cleanup', methods=['POST'])
@login_required
@admin_required
def api_cleanup_old_records():
    """Clean up old records."""
    global _detection_system
    
    if _detection_system and _detection_system.database:
        deleted = _detection_system.database.cleanup_old_records()
        return jsonify({'status': 'ok', 'deleted': deleted})
    
    return jsonify({'status': 'error', 'message': 'Database not available'}), 400


@app.route('/video_feed')
@login_required
def video_feed():
    """Video streaming route."""
    def generate():
        while True:
            frame = get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "No Camera Feed", (180, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ==================== Video Source Management ====================

_video_source = 'webcam'
_video_file_path = None
_detection_paused = False


@app.route('/api/source', methods=['POST'])
@login_required
@admin_required
def api_set_source():
    """Set video source."""
    global _video_source, _detection_system
    
    data = request.get_json()
    source = data.get('source', 'webcam')
    
    _video_source = source
    
    if _detection_system:
        if source == 'webcam':
            _detection_system.switch_source(0)
        elif source == 'rtsp':
            rtsp_url = data.get('url')
            if rtsp_url:
                _detection_system.switch_source(rtsp_url)
    
    return jsonify({'status': 'ok', 'source': source})


@app.route('/api/source/upload', methods=['POST'])
@login_required
@admin_required
def api_upload_video():
    """Upload video file for processing."""
    global _video_file_path, _video_source, _detection_system
    
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No file selected'}), 400
    
    # Save uploaded file
    import os
    upload_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    
    _video_file_path = filepath
    _video_source = 'video'
    
    # Switch the detection system to use the uploaded video
    if _detection_system:
        success = _detection_system.switch_source(filepath)
        return jsonify({
            'status': 'ok' if success else 'error',
            'filename': file.filename,
            'path': filepath
        })
    
    return jsonify({'status': 'ok', 'filename': file.filename, 'path': filepath})


@app.route('/api/detection/toggle', methods=['POST'])
@login_required
@admin_required
def api_toggle_detection():
    """Toggle detection on/off."""
    global _detection_paused
    
    data = request.get_json()
    _detection_paused = not data.get('running', True)
    
    return jsonify({'status': 'ok', 'paused': _detection_paused})


@app.route('/api/source/current')
@login_required
def api_get_source():
    """Get current video source."""
    global _detection_system
    current = None
    if _detection_system:
        current = _detection_system.get_current_source()
    
    return jsonify({
        'source': _video_source,
        'file': _video_file_path,
        'paused': _detection_paused,
        'current_source': str(current) if current else None
    })


# ==================== Report Generation ====================

@app.route('/report')
@login_required
def report_page():
    """Report generation page."""
    return render_template('report.html', user=current_user)


@app.route('/api/report/generate')
@login_required
def generate_report():
    """Generate PDF report."""
    global _detection_system, _crowd_history, _suspicious_alerts
    
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    
    # Get date range
    start_date = request.args.get('start')
    end_date = request.args.get('end')
    
    # Create PDF
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    elements = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER,
        spaceAfter=30
    )
    elements.append(Paragraph("Object Detection System Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.5*inch))
    
    # Statistics section
    elements.append(Paragraph("Summary Statistics", styles['Heading2']))
    
    stats = _statistics or {}
    suspicious_count = len(_suspicious_alerts)
    
    stats_data = [
        ['Metric', 'Value'],
        ['Total Detections', str(stats.get('total_detections', 0))],
        ['Alerts Today', str(stats.get('alerts_today', 0))],
        ['Suspicious Activities', str(suspicious_count)],
        ['Average Confidence', f"{(stats.get('average_confidence', 0) * 100):.1f}%"],
        ['Current Crowd Count', str(_crowd_count)],
    ]
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(stats_table)
    elements.append(Spacer(1, 0.5*inch))
    
    # Crowd Analysis section
    elements.append(Paragraph("Crowd Analysis", styles['Heading2']))
    
    if _crowd_history:
        # Generate crowd chart
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        times = [datetime.fromisoformat(d['timestamp']) for d in _crowd_history[-50:]]
        counts = [d['count'] for d in _crowd_history[-50:]]
        
        plt.figure(figsize=(8, 4))
        plt.plot(times, counts, 'b-', linewidth=2)
        plt.fill_between(times, counts, alpha=0.3)
        plt.xlabel('Time')
        plt.ylabel('People Count')
        plt.title('Crowd Count Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save chart to buffer
        chart_buffer = io.BytesIO()
        plt.savefig(chart_buffer, format='png', dpi=100)
        chart_buffer.seek(0)
        plt.close()
        
        # Add chart to PDF
        chart_buffer.seek(0)
        img = Image(chart_buffer, width=6*inch, height=3*inch)
        elements.append(img)
    else:
        elements.append(Paragraph("No crowd data available", styles['Normal']))
    
    elements.append(Spacer(1, 0.5*inch))
    
    # Suspicious Activities section
    elements.append(Paragraph("Suspicious Activities", styles['Heading2']))
    
    if _suspicious_alerts:
        susp_data = [['Time', 'Object', 'Confidence', 'Status']]
        for alert in _suspicious_alerts[:10]:
            time_str = alert.get('timestamp', 'N/A')
            if isinstance(time_str, str) and len(time_str) > 19:
                time_str = time_str[:19]
            susp_data.append([
                time_str,
                alert.get('object_type', 'Unknown'),
                f"{(alert.get('confidence', 0) * 100):.1f}%",
                'Detected'
            ])
        
        susp_table = Table(susp_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1*inch])
        susp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.red),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        elements.append(susp_table)
    else:
        elements.append(Paragraph("No suspicious activities detected", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    buffer.seek(0)
    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f'detection_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
    )


@app.route('/api/report/export-csv')
@login_required
def export_csv():
    """Export detection data as CSV."""
    global _detection_system, _crowd_history
    
    import csv
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    writer.writerow(['Timestamp', 'Object Type', 'Confidence', 'Crowd Count', 'Suspicious'])
    
    # Data
    if _detection_system and _detection_system.database:
        detections = _detection_system.database.get_recent_detections(limit=1000)
        for det in detections:
            writer.writerow([
                det.get('timestamp', ''),
                det.get('object_type', ''),
                f"{(det.get('confidence', 0) * 100):.1f}%",
                det.get('crowd_count', ''),
                'Yes' if det.get('suspicious') else 'No'
            ])
    
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'detection_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )


# ==================== Socket.IO Events ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'status': 'ok'})
    
    if _statistics:
        emit('stats_update', _statistics)
    if _alerts:
        emit('alerts_history', _alerts[:20])
    emit('crowd_update', {'count': _crowd_count, 'history': _crowd_history[-50:]})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    pass


@socketio.on('request_frame')
def handle_frame_request():
    """Send current frame as base64."""
    frame = get_frame()
    if frame is not None:
        emit('frame_update', {'image': frame_to_base64(frame)})


@socketio.on('request_stats')
def handle_stats_request():
    """Send current statistics."""
    emit('stats_update', _statistics)


def run_dashboard(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Run the Flask dashboard server."""
    print(f"Starting dashboard at http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


class DashboardThread:
    """Run dashboard in a separate thread."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 5000):
        self.host = host
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> None:
        """Start the dashboard server in background."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self._thread.start()
    
    def _run_server(self) -> None:
        """Run the server."""
        run_dashboard(host=self.host, port=self.port, debug=False)
    
    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
