"""
DeepSORT Tracker Module
Implements multi-object tracking using DeepSORT algorithm for smooth
object tracking in crowded scenes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
import cv2


@dataclass
class Track:
    """
    Represents a single tracked object.
    
    Attributes:
        track_id: Unique identifier for this track
        bbox: Current bounding box [x1, y1, x2, y2]
        class_name: Object class name
        confidence: Detection confidence
        age: Number of frames since track was created
        hits: Number of successful detections
        missed: Number of consecutive missed detections
        center_history: List of recent center positions
        velocity: Estimated velocity [vx, vy]
    """
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 1
    missed: int = 0
    center_history: List[Tuple[int, int]] = field(default_factory=list)
    velocity: Tuple[float, float] = (0.0, 0.0)
    
    def get_center(self) -> Tuple[int, int]:
        """Get center of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def get_area(self) -> int:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def predict_bbox(self) -> Tuple[int, int, int, int]:
        """Predict next bounding box position using velocity."""
        x1, y1, x2, y2 = self.bbox
        vx, vy = self.velocity
        
        # Predict new position
        new_x1 = int(x1 + vx)
        new_y1 = int(y1 + vy)
        new_x2 = int(x2 + vx)
        new_y2 = int(y2 + vy)
        
        return (new_x1, new_y1, new_x2, new_y2)
    
    def update_velocity(self, new_center: Tuple[int, int], alpha: float = 0.7):
        """Update velocity estimate using exponential moving average."""
        if self.center_history:
            old_center = self.center_history[-1]
            new_vx = new_center[0] - old_center[0]
            new_vy = new_center[1] - old_center[1]
            
            # Smooth velocity update
            vx, vy = self.velocity
            self.velocity = (
                alpha * new_vx + (1 - alpha) * vx,
                alpha * new_vy + (1 - alpha) * vy
            )


class KalmanBoxTracker:
    """
    Kalman filter for bounding box tracking.
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self, bbox: Tuple[int, int, int, int]):
        """Initialize Kalman filter with initial bounding box."""
        self.kf = self._init_kalman_filter()
        
        # Initialize state with bounding box
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        # Use statePost for initial state (OpenCV KalmanFilter API)
        self.kf.statePost[:4, 0] = np.array([cx, cy, w, h], dtype=np.float32)
        self.time_since_update = 0
        self.id = id(self)
        self.history = []
        
    def _init_kalman_filter(self) -> cv2.KalmanFilter:
        """Initialize OpenCV Kalman filter."""
        kf = cv2.KalmanFilter(8, 4)
        
        # State transition matrix
        kf.transitionMatrix = np.eye(8, dtype=np.float32)
        for i in range(4):
            kf.transitionMatrix[i, i + 4] = 1
        
        # Measurement matrix
        kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            kf.measurementMatrix[i, i] = 1
        
        # Process noise covariance
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        for i in range(4):
            kf.processNoiseCov[i, i] = 1e-1
        
        # Measurement noise covariance
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        
        # Initialize statePost
        kf.statePost = np.zeros((8, 1), dtype=np.float32)
        
        return kf
    
    def predict(self) -> Tuple[int, int, int, int]:
        """Predict next state."""
        prediction = self.kf.predict()
        cx, cy, w, h = prediction[:4, 0]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return (int(x1), int(y1), int(x2), int(y2))
    
    def update(self, bbox: Tuple[int, int, int, int]):
        """Update state with measurement."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        measurement = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(measurement)
        self.time_since_update = 0
    
    def get_state(self) -> Tuple[int, int, int, int]:
        """Get current state as bounding box."""
        cx, cy, w, h = self.kf.statePost[:4, 0]
        
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        
        return (int(x1), int(y1), int(x2), int(y2))


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker implementation.
    Uses Kalman filtering and Hungarian algorithm for data association.
    """
    
    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        max_distance: float = 100.0
    ):
        """
        Initialize DeepSORT tracker.
        
        Args:
            max_age: Maximum frames to keep track without detection
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            max_distance: Maximum distance for track-detection association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance
        
        self.tracks: List[Track] = []
        self.kalman_trackers: Dict[int, KalmanBoxTracker] = {}
        self._next_id = 1
        self._frame_count = 0
        
        # Track history for analytics
        self.track_history: Dict[int, List[Tuple[int, int]]] = {}
        
    def _get_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_distance(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate distance between box centers."""
        cx1 = (box1[0] + box1[2]) / 2
        cy1 = (box1[1] + box1[3]) / 2
        cx2 = (box2[0] + box2[2]) / 2
        cy2 = (box2[1] + box2[3]) / 2
        
        return np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    def _associate_detections(self, detections: List[Any]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections with existing tracks using Hungarian algorithm.
        
        Returns:
            Tuple of (matches, unmatched_detections, unmatched_tracks)
        """
        if not self.tracks:
            return [], list(range(len(detections))), []
        
        if not detections:
            return [], [], list(range(len(self.tracks)))
        
        # Build cost matrix
        num_tracks = len(self.tracks)
        num_dets = len(detections)
        
        cost_matrix = np.zeros((num_dets, num_tracks))
        
        for d_idx, det in enumerate(detections):
            det_bbox = det.bbox
            
            for t_idx, track in enumerate(self.tracks):
                # Predicted position
                pred_bbox = track.predict_bbox() if track.center_history else track.bbox
                
                # Combined cost: IoU and distance
                iou = self._get_iou(det_bbox, pred_bbox)
                dist = self._get_distance(det_bbox, pred_bbox)
                
                # Cost: lower is better
                # Use negative IoU (since we want to maximize it)
                # and add normalized distance
                cost_matrix[d_idx, t_idx] = -iou + (dist / self.max_distance) * 0.5
        
        # Hungarian algorithm for assignment
        det_indices, track_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_dets = list(range(num_dets))
        unmatched_tracks = list(range(num_tracks))
        
        for d_idx, t_idx in zip(det_indices, track_indices):
            # Check if match is valid
            det = detections[d_idx]
            track = self.tracks[t_idx]
            
            iou = self._get_iou(det.bbox, track.bbox)
            dist = self._get_distance(det.bbox, track.bbox)
            
            if iou >= self.iou_threshold or dist <= self.max_distance:
                matches.append((d_idx, t_idx))
                unmatched_dets.remove(d_idx)
                unmatched_tracks.remove(t_idx)
        
        return matches, unmatched_dets, unmatched_tracks
    
    def update(self, detections: List[Any]) -> List[Track]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            List of active, confirmed tracks
        """
        self._frame_count += 1
        
        # Associate detections with existing tracks
        matches, unmatched_dets, unmatched_tracks = self._associate_detections(detections)
        
        # Update matched tracks
        for d_idx, t_idx in matches:
            det = detections[d_idx]
            track = self.tracks[t_idx]
            
            # Update track
            old_center = track.get_center()
            track.bbox = det.bbox
            track.confidence = det.confidence
            track.age += 1
            track.hits += 1
            track.missed = 0
            
            # Update velocity
            new_center = track.get_center()
            track.update_velocity(new_center)
            
            # Update center history
            track.center_history.append(new_center)
            if len(track.center_history) > 30:  # Keep last 30 positions
                track.center_history.pop(0)
            
            # Update Kalman filter
            if track.track_id in self.kalman_trackers:
                self.kalman_trackers[track.track_id].update(det.bbox)
        
        # Handle unmatched tracks
        for t_idx in unmatched_tracks:
            track = self.tracks[t_idx]
            track.missed += 1
            track.age += 1
            
            # Predict next position using Kalman filter
            if track.track_id in self.kalman_trackers:
                predicted_bbox = self.kalman_trackers[track.track_id].predict()
                track.bbox = predicted_bbox
                self.kalman_trackers[track.track_id].time_since_update += 1
        
        # Create new tracks for unmatched detections
        for d_idx in unmatched_dets:
            det = detections[d_idx]
            
            new_track = Track(
                track_id=self._next_id,
                bbox=det.bbox,
                class_name=det.class_name,
                confidence=det.confidence,
                age=1,
                hits=1,
                missed=0,
                center_history=[((det.bbox[0] + det.bbox[2]) // 2, 
                                (det.bbox[1] + det.bbox[3]) // 2)]
            )
            
            # Initialize Kalman filter for new track
            self.kalman_trackers[self._next_id] = KalmanBoxTracker(det.bbox)
            
            self.tracks.append(new_track)
            self.track_history[self._next_id] = []
            self._next_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.missed <= self.max_age]
        
        # Update track history
        for track in self.tracks:
            if track.track_id in self.track_history:
                self.track_history[track.track_id].append(track.get_center())
                # Keep last 100 positions
                if len(self.track_history[track.track_id]) > 100:
                    self.track_history[track.track_id].pop(0)
        
        # Return confirmed tracks
        confirmed_tracks = [
            t for t in self.tracks 
            if t.hits >= self.min_hits and t.missed == 0
        ]
        
        return confirmed_tracks
    
    def get_all_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [t for t in self.tracks if t.missed < self.max_age]
    
    def get_track_count(self) -> Dict[str, int]:
        """Get count of tracked objects by class."""
        counts = {}
        for track in self.get_all_active_tracks():
            counts[track.class_name] = counts.get(track.class_name, 0) + 1
        return counts
    
    def get_person_tracks(self) -> List[Track]:
        """Get all person tracks."""
        return [t for t in self.get_all_active_tracks() if t.class_name.lower() == 'person']
    
    def draw_tracks(
        self, 
        frame: np.ndarray, 
        draw_trajectory: bool = True,
        draw_id: bool = True,
        color_by_id: bool = True
    ) -> np.ndarray:
        """
        Draw tracks on frame.
        
        Args:
            frame: Input frame
            draw_trajectory: Whether to draw trajectory line
            draw_id: Whether to draw track ID
            color_by_id: Whether to color boxes by track ID
            
        Returns:
            Frame with tracks drawn
        """
        output = frame.copy()
        
        for track in self.get_all_active_tracks():
            x1, y1, x2, y2 = track.bbox
            
            # Get color based on track ID or class
            if color_by_id:
                np.random.seed(track.track_id * 123)
                color = tuple(map(int, np.random.randint(50, 255, 3)))
            else:
                color = (0, 255, 0) if track.class_name.lower() == 'person' else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw trajectory
            if draw_trajectory and len(track.center_history) > 1:
                points = np.array(track.center_history, dtype=np.int32)
                cv2.polylines(output, [points], False, color, 1, cv2.LINE_AA)
            
            # Draw ID and class
            if draw_id:
                label = f"#{track.track_id} {track.class_name}"
                cv2.putText(output, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output
    
    def reset(self):
        """Reset tracker state."""
        self.tracks = []
        self.kalman_trackers = {}
        self._next_id = 1
        self._frame_count = 0
        self.track_history = {}


class CrowdTracker:
    """
    Specialized tracker for crowd analysis.
    Tracks people and provides crowd-specific analytics.
    """
    
    def __init__(self, tracker: Optional[DeepSORTTracker] = None):
        """Initialize crowd tracker."""
        self.tracker = tracker or DeepSORTTracker()
        self._entry_count = 0
        self._exit_count = 0
        self._total_entries = 0
        self._total_exits = 0
        
    def update(self, detections: List[Any], frame_shape: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Update crowd tracking.
        
        Args:
            detections: List of Detection objects
            frame_shape: Frame shape (height, width) for entry/exit detection
            
        Returns:
            Dictionary with crowd analytics
        """
        # Filter person detections
        person_detections = [d for d in detections if d.class_name.lower() == 'person']
        
        # Update tracker
        tracks = self.tracker.update(person_detections)
        person_tracks = [t for t in tracks if t.class_name.lower() == 'person']
        
        # Count entries and exits if frame shape is provided
        if frame_shape:
            height, width = frame_shape[:2]
            center_x = width // 2
            
            current_left = 0
            current_right = 0
            
            for track in person_tracks:
                cx, _ = track.get_center()
                if cx < center_x:
                    current_left += 1
                else:
                    current_right += 1
        
        return {
            'total_people': len(person_tracks),
            'tracks': person_tracks,
            'entry_count': self._entry_count,
            'exit_count': self._exit_count,
            'avg_confidence': np.mean([t.confidence for t in person_tracks]) if person_tracks else 0,
            'track_ids': [t.track_id for t in person_tracks]
        }
    
    def get_tracker(self) -> DeepSORTTracker:
        """Get underlying DeepSORT tracker."""
        return self.tracker
    
    def draw_crowd_info(
        self,
        frame: np.ndarray,
        show_trajectories: bool = True
    ) -> np.ndarray:
        """Draw crowd tracking info on frame."""
        output = self.tracker.draw_tracks(frame, draw_trajectory=show_trajectories)
        
        # Add crowd count
        person_count = len(self.tracker.get_person_tracks())
        cv2.putText(output, f"Crowd: {person_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return output
    
    def reset(self):
        """Reset crowd tracker."""
        self.tracker.reset()
        self._entry_count = 0
        self._exit_count = 0
