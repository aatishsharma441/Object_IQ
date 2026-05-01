"""
Dashboard Launcher - Enhanced Version with DeepSORT Tracking
Run the object detection system with web dashboard, crowd counting,
suspicious activity detection, and smooth multi-object tracking.
"""

import signal
import sys
import threading
import time
import base64
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.detection import ObjectDetector
from src.detection.tracker import DeepSORTTracker, CrowdTracker
from src.detection.weapon_detector import WeaponDetector
from src.detection.heatmap import CrowdAnalytics, HeatmapConfig
import src.logic
from src.notifications import EmailService
from src.notifications.email_service import EmailConfig
from src.storage import Database, SnapshotManager
from src.utils import CameraManager
from src.utils.logger import DetectionLogger
from src.web import (
    set_detection_system,
    update_frame,
    add_alert,
    update_statistics,
    add_crowd_data,
    run_dashboard
)

# Suspicious objects to detect (weapons, etc.)
SUSPICIOUS_OBJECTS = ['gun', 'knife', 'weapon', 'scissors', 'baseball bat', 'stone', 'rock', 'pistol', 'rifle']
# Minimum confidence threshold for suspicious detection
SUSPICIOUS_CONFIDENCE_THRESHOLD = 0.50


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value (0-1)
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def calculate_proximity(box1, box2):
    """
    Calculate proximity between two bounding boxes.
    Returns distance between centers normalized by diagonal.
    """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    # Normalize by frame diagonal
    return distance


def get_severity_level(confidence: float, object_types: list, person_nearby: bool) -> str:
    """
    Determine severity level based on confidence and context.
    
    Returns: 'low', 'medium', or 'high'
    """
    # High threat: weapon + person nearby + high confidence
    if person_nearby and confidence >= 0.85:
        return 'high'
    # Medium threat: weapon detected with moderate confidence or person nearby
    elif person_nearby or confidence >= 0.80:
        return 'medium'
    # Low threat: basic detection
    else:
        return 'low'


class DashboardDetectionSystem:
    """
    Object detection system with integrated web dashboard.
    Includes crowd counting and suspicious activity detection.
    """
    
    def __init__(self, config_path: Optional[str] = None, dashboard_port: int = 5000):
        """
        Initialize the detection system with dashboard.
        
        Args:
            config_path: Optional path to configuration file
            dashboard_port: Port for web dashboard
        """
        self.config = get_config(config_path)
        self.dashboard_port = dashboard_port
        
        # Initialize logger
        self.logger = DetectionLogger(
            name="object_detection",
            log_dir=self.config.log_dir,
            level=self.config.log_level
        )
        
        # Components
        self.detector: Optional[ObjectDetector] = None
        self.decision_engine: Optional[src.logic.DecisionEngine] = None
        self.camera: Optional[CameraManager] = None
        self.database: Optional[Database] = None
        self.snapshot_manager: Optional[SnapshotManager] = None
        self.email_service: Optional[EmailService] = None
        
        # State
        self._running = False
        self._frame_count = 0
        self._dashboard_thread: Optional[threading.Thread] = None
        self._last_crowd_update = 0
        self._source_lock = threading.Lock()
        self._pending_source = None
        
        # Frame optimization
        self._frame_skip_counter = 0
        self._frame_skip_interval = 2  # Process every nth frame
        self._processing_scale = 0.75  # Scale down for processing
        
        # Authentication state
        self._authenticated_users = set()  # Track logged-in users
        self._detection_enabled = True  # Requires authentication
        
        # Severity tracking
        self._severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        # Cooldown for suspicious alerts
        self._suspicious_cooldown = {}  # {object_type: last_alert_time}
        self._cooldown_seconds = 5  # Seconds between same type alerts
        
        # Crowd settings
        crowd_config = self.config._objects_config.get('crowd_detection', {})
        self.crowd_enabled = crowd_config.get('enabled', True)
        self.crowd_alert_threshold = crowd_config.get('alert_threshold', 5)
        self.crowd_high_threshold = crowd_config.get('high_density_threshold', 10)
        
        # Initialize DeepSORT tracker for smooth tracking
        self.tracker = DeepSORTTracker(
            max_age=30,
            min_hits=3,
            iou_threshold=0.3
        )
        
        # Initialize crowd tracker
        self.crowd_tracker = CrowdTracker(self.tracker)
        
        # Initialize weapon detector for improved gun/knife detection
        self.weapon_detector = WeaponDetector(
            confidence_threshold=SUSPICIOUS_CONFIDENCE_THRESHOLD
        )
        
        # Initialize crowd analytics and heatmap
        heatmap_config = HeatmapConfig(
            grid_size=20,
            sigma=15.0,
            history_length=100,
            decay_rate=0.95
        )
        self.crowd_analytics = CrowdAnalytics(history_length=100)
        
        # Heatmap data for API
        self._heatmap_data = {'image': None, 'analysis': {}}
        
        # Track ID colors for consistent visualization
        self._track_colors = {}
    
    def initialize(self) -> bool:
        """Initialize all system components."""
        try:
            self.logger.log_startup({
                "Model": self.config.model_name,
                "Resolution": self.config.resolution,
                "Frame Rate": self.config.frame_rate,
                "Target Objects": self.config.target_objects,
                "Confidence Threshold": self.config.confidence_threshold,
                "Temporal Filter": f"{self.config.consecutive_frames} frames",
                "Cooldown": f"{self.config.cooldown_seconds}s",
                "ROI Enabled": self.config.roi_enabled,
                "Dashboard Port": self.dashboard_port,
                "Crowd Detection": "Enabled" if self.crowd_enabled else "Disabled"
            })
            
            # Initialize detector with ALL objects (YOLO detects all, we filter later)
            self.logger.logger.info("Initializing object detector...")
            self.detector = ObjectDetector(
                model_name=self.config.model_name,
                model_dir=Path(__file__).parent.parent / "models",
                use_onnx=self.config.use_onnx,
                confidence_threshold=self.config.confidence_threshold,
                target_objects=None,  # Detect all objects
                resolution=self.config.resolution
            )
            self.detector.initialize()
            
            # Initialize decision engine
            self.logger.logger.info("Initializing decision engine...")
            self.decision_engine = src.logic.DecisionEngine(
                target_objects=self.config.target_objects,
                confidence_threshold=self.config.confidence_threshold,
                consecutive_frames=self.config.consecutive_frames,
                cooldown_seconds=self.config.cooldown_seconds,
                roi_enabled=self.config.roi_enabled,
                roi_zones=self.config.roi_zones,
                object_config=self.config.object_config
            )
            
            # Initialize camera
            self.logger.logger.info("Initializing camera...")
            self.camera = CameraManager(
                source=self.config.camera_source,
                reconnect_delay=self.config.reconnect_delay,
                frame_rate=self.config.frame_rate,
                resolution=self.config.resolution
            )
            
            # Initialize database
            self.logger.logger.info("Initializing MongoDB Atlas database...")
            self.database = Database(
                connection_string=self.config.mongodb_connection_string,
                database_name=self.config.mongodb_database_name,
                collections=self.config.mongodb_collections,
                options=self.config.mongodb_options,
                retention_days=self.config.database.get('retention_days', 30),
                cache_enabled=self.config.cache_enabled,
                cache_max_size_mb=self.config.cache_max_size_mb
            )
            
            db_success, db_message = self.database.test_connection()
            if db_success:
                self.logger.logger.info("MongoDB Atlas connected successfully")
            else:
                self.logger.logger.warning(f"MongoDB Atlas connection failed: {db_message}")
            
            # Initialize snapshot manager
            self.logger.logger.info("Initializing snapshot manager...")
            self.snapshot_manager = SnapshotManager(
                snapshot_dir=str(self.config.snapshot_dir),
                max_age_days=self.config.logging_config.get('max_snapshot_days', 7)
            )
            
            # Initialize email service
            if self.config.email_enabled:
                self.logger.logger.info("Initializing email service...")
                email_config = EmailConfig(
                    smtp_server=self.config.email_config.get('smtp_server', 'smtp.gmail.com'),
                    smtp_port=self.config.email_config.get('smtp_port', 587),
                    sender_email=self.config.email_config.get('sender_email', ''),
                    sender_password=self.config.email_config.get('sender_password', ''),
                    recipient_email=self.config.email_config.get('recipient_email', ''),
                    use_tls=self.config.email_config.get('use_tls', True)
                )
                self.email_service = EmailService(email_config)
                
                success, message = self.email_service.test_connection()
                if success:
                    self.logger.logger.info("Email service connected successfully")
                else:
                    self.logger.logger.warning(f"Email service test failed: {message}")
            
            # Set detection system reference for dashboard
            set_detection_system(self)
            
            self.logger.logger.info("System initialization complete")
            return True
            
        except Exception as e:
            self.logger.log_error("initialization", e)
            return False
    
    def start_dashboard(self) -> None:
        """Start the web dashboard in a background thread."""
        self._dashboard_thread = threading.Thread(
            target=run_dashboard,
            kwargs={'host': '0.0.0.0', 'port': self.dashboard_port},
            daemon=True
        )
        self._dashboard_thread.start()
        self.logger.logger.info(f"Dashboard started at http://localhost:{self.dashboard_port}")
    
    def run(self) -> None:
        """Run the main detection loop with dashboard."""
        if not self.initialize():
            self.logger.logger.error("Failed to initialize system")
            return
        
        # Start dashboard
        self.start_dashboard()
        
        # Connect to camera
        if not self.camera.connect():
            self.logger.logger.error("Failed to connect to camera")
            return
        
        self._running = True
        self.logger.logger.info("Starting detection loop...")
        
        frame_interval = 1.0 / self.config.frame_rate
        stats_update_interval = 5.0
        last_stats_update = time.time()
        
        try:
            while self._running:
                start_time = time.time()
                
                # Check for pending source switch
                with self._source_lock:
                    if self._pending_source is not None:
                        self.logger.logger.info(f"Switching to source: {self._pending_source}")
                        self.camera.disconnect()
                        self.camera.source = self._pending_source
                        if self.camera.connect():
                            self.logger.logger.info("Source switch successful")
                        else:
                            self.logger.logger.error("Source switch failed")
                        self._pending_source = None
                
                # Read frame
                success, frame = self.camera.read_frame()
                
                if not success or frame is None:
                    self.logger.log_camera_event("frame_error", "Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                self._frame_count += 1
                
                # Process frame
                annotated_frame, crowd_count = self._process_frame(frame)
                
                # Update dashboard frame with crowd count
                update_frame(annotated_frame, crowd_count=crowd_count)
                
                # Update crowd data periodically
                if time.time() - self._last_crowd_update >= 2.0:
                    add_crowd_data(crowd_count)
                    self._last_crowd_update = time.time()
                
                # Update statistics periodically
                if time.time() - last_stats_update >= stats_update_interval:
                    self._update_dashboard_stats()
                    last_stats_update = time.time()
                
                # Maintain frame rate
                elapsed = time.time() - start_time
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except KeyboardInterrupt:
            self.logger.logger.info("Received keyboard interrupt")
        finally:
            self.shutdown()
    
    def _process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a frame and return annotated version and crowd count.
        Uses DeepSORT for smooth multi-object tracking.
        
        Returns:
            Tuple of (annotated_frame, crowd_count)
        """
        # Run detection on ALL objects
        all_detections = self.detector.detect(frame)
        
        # Frame optimization: skip processing for some frames
        self._frame_skip_counter += 1
        
        # Separate people and other objects
        person_detections = [d for d in all_detections if d.class_name.lower() == 'person']
        
        # Apply DeepSORT tracking for smooth person tracking
        tracked_persons = self.tracker.update(person_detections)
        crowd_count = len(tracked_persons)
        
        # Update crowd analytics and heatmap
        crowd_analytics_data = self.crowd_analytics.update(all_detections, frame)
        self._heatmap_data['analysis'] = crowd_analytics_data
        
        # Generate heatmap periodically
        if self._frame_count % 30 == 0:  # Every 30 frames
            heatmap_img = self.crowd_analytics.get_visualization(frame, show_heatmap=True)
            _, buffer = cv2.imencode('.jpg', heatmap_img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self._heatmap_data['image'] = base64.b64encode(buffer).decode('utf-8')
        
        # Get person bounding boxes for proximity check
        person_boxes = [t.bbox for t in tracked_persons]
        
        # Use weapon detector for improved weapon detection
        normal_detections, weapon_detections = self.weapon_detector.filter_detections(all_detections, frame)
        
        # Handle weapon detections
        for wdet in weapon_detections:
            if not wdet.is_weapon:
                continue
            
            # Create detection wrapper
            class WeaponWrapper:
                def __init__(self, wdet):
                    self.class_name = wdet.weapon_type or wdet.class_name
                    self.confidence = wdet.confidence
                    self.bbox = wdet.bbox
                    self.center = ((wdet.bbox[0] + wdet.bbox[2]) // 2, (wdet.bbox[1] + wdet.bbox[3]) // 2)
            
            det = WeaponWrapper(wdet)
            
            # Check if person is nearby
            person_nearby = False
            if det.bbox and person_boxes:
                for pbox in person_boxes:
                    iou = calculate_iou(det.bbox, pbox)
                    proximity = calculate_proximity(det.bbox, pbox)
                    if iou > 0.1 or proximity < 200:
                        person_nearby = True
                        break
            
            severity = wdet.threat_level
            
            # Check cooldown
            obj_key = f"{det.class_name}_{severity}"
            now = time.time()
            if obj_key in self._suspicious_cooldown:
                if now - self._suspicious_cooldown[obj_key] < self._cooldown_seconds:
                    continue
            
            self._suspicious_cooldown[obj_key] = now
            
            # Handle weapon detection
            self._handle_weapon_detection(frame, det, severity, person_nearby, crowd_count)
        
        
        # Also check remaining suspicious objects
        suspicious_detections = [
            d for d in normal_detections 
            if d.class_name.lower() in SUSPICIOUS_OBJECTS
            and d.confidence >= SUSPICIOUS_CONFIDENCE_THRESHOLD
        ]
        
        # Handle suspicious activity
        for det in suspicious_detections:
            person_nearby = False
            
            if det.bbox and person_boxes:
                for pbox in person_boxes:
                    iou = calculate_iou(det.bbox, pbox)
                    proximity = calculate_proximity(det.bbox, pbox)
                    if iou > 0.1 or proximity < 200:
                        person_nearby = True
                        break
            
            severity = get_severity_level(det.confidence, [det.class_name], person_nearby)
            
            obj_key = f"{det.class_name}_{severity}"
            now = time.time()
            if obj_key in self._suspicious_cooldown:
                if now - self._suspicious_cooldown[obj_key] < self._cooldown_seconds:
                    continue
            
            self._suspicious_cooldown[obj_key] = now
            
            
            self._handle_suspicious_activity(frame, det, severity, person_nearby, crowd_count)
        
        
        # Filter detections for normal alerts
        normal_detections = [
            d for d in normal_detections 
            if d.class_name.lower() in [t.lower() for t in self.config.target_objects]
            and d.class_name.lower() not in SUSPICIOUS_OBJECTS
        ]
        
        # Draw tracked detections on frame
        annotated_frame = self._draw_tracked_frame(frame, tracked_persons, all_detections, weapon_detections)
        
        # Add crowd count overlay
        cv2.putText(
            annotated_frame, 
            f"People: {crowd_count}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0) if crowd_count < self.crowd_alert_threshold else (0, 0, 255),
            2
        )
        
        # Add tracking info
        cv2.putText(
            annotated_frame,
            f"Tracking: {len(tracked_persons)} IDs",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )
        
        # Process normal detections through decision engine
        if normal_detections:
            det_dicts = [d.to_dict() for d in normal_detections]
            det_dicts = [d | {'crowd_count': crowd_count} for d in det_dicts]
            
            decisions = self.decision_engine.process_detections(det_dicts)
            
            for decision in decisions:
                if decision.should_alert:
                    self._handle_alert(frame, decision, crowd_count)
        
        
        # Crowd alert
        if crowd_count >= self.crowd_high_threshold:
            self._handle_crowd_alert(frame, crowd_count, 'high')
        elif crowd_count >= self.crowd_alert_threshold:
            self._handle_crowd_alert(frame, crowd_count, 'moderate')
        
        return annotated_frame, crowd_count
    
    def _draw_tracked_frame(self, frame: np.ndarray, tracked_persons: List, all_detections: List, weapon_detections: List) -> np.ndarray:
        """Draw frame with tracked objects and trajectories."""
        output = frame.copy()
        
        # Draw tracked persons with trajectories
        for track in tracked_persons:
            x1, y1, x2, y2 = track.bbox
            
            # Get consistent color for track ID
            if track.track_id not in self._track_colors:
                np.random.seed(track.track_id * 123)
                self._track_colors[track.track_id] = tuple(map(int, np.random.randint(50, 255, 3)))
            color = self._track_colors[track.track_id]
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw trajectory
            if len(track.center_history) > 1:
                points = np.array(track.center_history, dtype=np.int32)
                cv2.polylines(output, [points], False, color, 1, cv2.LINE_AA)
            
            # Draw ID and class
            label = f"#{track.track_id} Person"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw weapon detections
        for wdet in weapon_detections:
            if wdet.is_weapon:
                output = self.weapon_detector.draw_weapon_detections(output, [wdet])
        
        # Draw other detections
        for det in all_detections:
            if det.class_name.lower() == 'person':
                continue
            
            x1, y1, x2, y2 = det.bbox
            class_name = det.class_name
            
            if class_name.lower() in SUSPICIOUS_OBJECTS:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            cv2.putText(output, f"{class_name} {det.confidence:.0%}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output
    
    def _handle_weapon_detection(self, frame: np.ndarray, detection, severity: str, person_nearby: bool, crowd_count: int) -> None:
        """Handle weapon detection with email alert."""
        timestamp = datetime.now()
        
        self._severity_counts[severity] = self._severity_counts.get(severity, 0) + 1
        
        severity_icon = {'high': '🚨', 'medium': '⚠️', 'low': '⚠'}
        self.logger.logger.warning(
            f"{severity_icon.get(severity, '⚠')} WEAPON [{severity.upper()}]: {detection.class_name} detected "
            f"with {detection.confidence:.1%} confidence! Person nearby: {person_nearby}"
        )
        
        # Save snapshot
        snapshot_path = None
        if self.snapshot_manager:
            snapshot_path = self.snapshot_manager.save_snapshot(
                frame=frame,
                object_type=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                timestamp=timestamp
            )
        
        # Store image in database as base64
        image_base64 = None
        if snapshot_path:
            try:
                with open(snapshot_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
            except:
                pass
        
        # Log to database
        detection_id = None
        if self.database:
            detection_id = self.database.log_detection(
                object_type=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                frame_number=self._frame_count,
                snapshot_path=snapshot_path,
                notified=True,
                metadata={
                    'suspicious': True,
                    'weapon': True,
                    'image_base64': image_base64,
                    'severity': severity,
                    'person_nearby': person_nearby,
                    'crowd_count': crowd_count
                }
            )
        
        # Send email alert for weapons ONLY
        email_sent = False
        if self.config.email_enabled and self.email_service:
            severity_subject = {'high': '[URGENT]', 'medium': '[WARNING]', 'low': '[ALERT]'}
            success, message = self.email_service.send_alert(
                object_type=f"{severity_subject.get(severity, '[ALERT]')} WEAPON: {detection.class_name}",
                confidence=detection.confidence,
                timestamp=timestamp,
                consecutive_frames=1,
                snapshot_path=snapshot_path
            )
            email_sent = success
            if success:
                self.logger.logger.info(f"Weapon alert email sent successfully")
            else:
                self.logger.logger.warning(f"Failed to send weapon alert email: {message}")
                
        # Log alert status
        notif_status = "sent" if email_sent else "skipped"
        self.logger.logger.warning(
            f"WEAPON ALERT: {detection.class_name} detected | Email {notif_status}"
        )
        
        # Add to dashboard alerts
        add_alert({
            'object_type': f"{severity_icon.get(severity, '⚠')} {detection.class_name.upper()}",
            'confidence': detection.confidence,
            'consecutive_frames': 1,
            'timestamp': timestamp.isoformat(),
            'suspicious': True,
            'weapon': True,
            'snapshot_path': snapshot_path,
            'severity': severity,
            'person_nearby': person_nearby,
            'crowd_count': crowd_count
        })
    
    def _handle_suspicious_activity(self, frame: np.ndarray, detection, severity: str = 'medium', person_nearby: bool = False, crowd_count: int = 0) -> None:
        """Handle suspicious activity detection (weapons, etc.)."""
        timestamp = datetime.now()
        
        # Update severity counts
        self._severity_counts[severity] = self._severity_counts.get(severity, 0) + 1
        
        # Log with severity level
        severity_icon = {'high': '🚨', 'medium': '⚠️', 'low': '⚠'}
        self.logger.logger.warning(
            f"{severity_icon.get(severity, '⚠')} THREAT [{severity.upper()}]: {detection.class_name} detected "
            f"with {detection.confidence:.1%} confidence! Person nearby: {person_nearby}"
        )
        
        # Save snapshot
        snapshot_path = None
        if self.snapshot_manager:
            snapshot_path = self.snapshot_manager.save_snapshot(
                frame=frame,
                object_type=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                timestamp=timestamp
            )
        
        # Store image in database as base64
        image_base64 = None
        if snapshot_path:
            try:
                with open(snapshot_path, 'rb') as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
            except:
                pass
        
        # Log to database with severity
        detection_id = None
        if self.database:
            detection_id = self.database.log_detection(
                object_type=detection.class_name,
                confidence=detection.confidence,
                bbox=detection.bbox,
                frame_number=self._frame_count,
                snapshot_path=snapshot_path,
                notified=True,
                metadata={
                    'suspicious': True, 
                    'image_base64': image_base64,
                    'severity': severity,
                    'person_nearby': person_nearby,
                    'crowd_count': crowd_count
                }
            )
        
        # Send immediate email alert
        email_sent = False
        if self.config.email_enabled and self.email_service:
            severity_subject = {'high': '[URGENT]', 'medium': '[WARNING]', 'low': '[ALERT]'}
            success, message = self.email_service.send_alert(
                object_type=f"{severity_subject.get(severity, '[ALERT]')} SUSPICIOUS: {detection.class_name}",
                confidence=detection.confidence,
                timestamp=timestamp,
                consecutive_frames=1,
                snapshot_path=snapshot_path
            )
            email_sent = success
            if success:
                self.logger.logger.info(f"Suspicious activity alert email sent successfully")
            else:
                self.logger.logger.warning(f"Failed to send suspicious activity alert email: {message}")
                
        # Log alert status
        notif_status = "sent" if email_sent else "skipped"
        self.logger.logger.warning(
            f"SUSPICIOUS ALERT: {detection.class_name} detected | Email {notif_status}"
        )
        
        # Add to dashboard alerts with severity
        add_alert({
            'object_type': f"{severity_icon.get(severity, '⚠')} {detection.class_name.upper()}",
            'confidence': detection.confidence,
            'consecutive_frames': 1,
            'timestamp': timestamp.isoformat(),
            'suspicious': True,
            'snapshot_path': snapshot_path,
            'severity': severity,
            'person_nearby': person_nearby,
            'crowd_count': crowd_count
        })
    
    def _handle_alert(self, frame: np.ndarray, decision, crowd_count: int) -> None:
        """Handle a normal alert decision."""
        self.logger.log_alert(
            object_type=decision.object_type,
            confidence=decision.confidence,
            consecutive_frames=decision.consecutive_frames,
            notification_sent=self.config.email_enabled
        )
        
        # Save snapshot
        snapshot_path = None
        if self.config.save_snapshots and self.snapshot_manager:
            snapshot_path = self.snapshot_manager.save_snapshot(
                frame=frame,
                object_type=decision.object_type,
                confidence=decision.confidence,
                bbox=decision.bbox,
                timestamp=decision.timestamp
            )
        
        # Log to database
        detection_id = None
        if self.database:
            detection_id = self.database.log_detection(
                object_type=decision.object_type,
                confidence=decision.confidence,
                bbox=decision.bbox,
                frame_number=self._frame_count,
                snapshot_path=snapshot_path,
                roi_zone=decision.roi_zone,
                notified=self.config.email_enabled,
                metadata={'crowd_count': crowd_count}
            )
        
        # Send notification
        if self.config.email_enabled and self.email_service:
            success, message = self.email_service.send_alert(
                object_type=decision.object_type,
                confidence=decision.confidence,
                timestamp=decision.timestamp,
                roi_zone=decision.roi_zone,
                consecutive_frames=decision.consecutive_frames,
                snapshot_path=snapshot_path
            )
            
            if self.database:
                self.database.log_alert(
                    detection_id=detection_id,
                    object_type=decision.object_type,
                    channel="email",
                    recipient=self.config.email_config.get('recipient_email', ''),
                    status="sent" if success else "failed",
                    error_message=None if success else message
                )
        
        # Update dashboard alert
        add_alert({
            'object_type': decision.object_type,
            'confidence': decision.confidence,
            'consecutive_frames': decision.consecutive_frames,
            'timestamp': decision.timestamp.isoformat() if decision.timestamp else None,
            'roi_zone': decision.roi_zone,
            'crowd_count': crowd_count,
            'suspicious': False
        })
    
    def _handle_crowd_alert(self, frame: np.ndarray, count: int, level: str) -> None:
        """Handle crowd density alert."""
        # Could send notification for crowd alerts if needed
        pass
    
    def _update_dashboard_stats(self) -> None:
        """Update dashboard statistics."""
        if self.database:
            stats = self.database.get_detection_stats()
            update_statistics(stats)
    
    def switch_source(self, source) -> bool:
        """
        Switch the video source dynamically.
        
        Args:
            source: New source (int for webcam, str for file/RTSP URL)
            
        Returns:
            True if switch was successful
        """
        with self._source_lock:
            self._pending_source = source
            self.logger.logger.info(f"Source switch requested: {source}")
            return True
    
    def get_current_source(self):
        """Get current video source."""
        if self.camera:
            return self.camera.source
        return None
    
    def shutdown(self) -> None:
        """Shutdown the system gracefully."""
        self.logger.log_shutdown()
        
        self._running = False
        
        if self.camera:
            self.camera.disconnect()
        
        if self.database:
            self.database.close()
        
        self.logger.logger.info("System shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Detection with Dashboard')
    parser.add_argument('--port', type=int, default=5000, help='Dashboard port')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    args = parser.parse_args()
    
    system = DashboardDetectionSystem(
        config_path=args.config,
        dashboard_port=args.port
    )
    system.run()


if __name__ == "__main__":
    main()
