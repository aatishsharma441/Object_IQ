"""
Temporal Filter Module
Implements consecutive frame detection filtering to reduce false positives.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TrackedObject:
    """
    Represents an object being tracked across frames.
    
    Attributes:
        class_name: Object class name
        consecutive_count: Number of consecutive frames detected
        last_bbox: Last known bounding box
        last_confidence: Last known confidence score
        last_center: Last known center point
    """
    class_name: str
    consecutive_count: int = 1
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_confidence: float = 0.0
    last_center: Optional[Tuple[int, int]] = None
    
    def update(
        self,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        center: Tuple[int, int]
    ) -> None:
        """Update tracked object with new detection."""
        self.consecutive_count += 1
        self.last_bbox = bbox
        self.last_confidence = confidence
        self.last_center = center
    
    def reset(self) -> None:
        """Reset consecutive count."""
        self.consecutive_count = 0


class TemporalFilter:
    """
    Filters detections based on consecutive frame requirements.
    Objects must be detected for N consecutive frames before being considered valid.
    This significantly reduces false positives from momentary detection errors.
    """
    
    def __init__(
        self,
        required_frames: int = 3,
        proximity_threshold: int = 50
    ):
        """
        Initialize temporal filter.
        
        Args:
            required_frames: Number of consecutive frames required
            proximity_threshold: Maximum distance (pixels) to consider
                               the same object between frames
        """
        self.required_frames = required_frames
        self.proximity_threshold = proximity_threshold
        
        # Tracking state: key = class_name, value = TrackedObject
        self._tracked_objects: Dict[str, TrackedObject] = {}
        
        # Track all objects seen in current frame for cleanup
        self._current_frame_objects: set = set()
    
    def process_frame(
        self,
        detections: List[dict]
    ) -> Tuple[List[dict], List[dict]]:
        """
        Process detections for a single frame.
        
        Args:
            detections: List of detection dictionaries with keys:
                       - class_name: str
                       - bbox: Tuple[int, int, int, int]
                       - confidence: float
                       - center: Tuple[int, int]
                       
        Returns:
            Tuple of (confirmed_detections, pending_detections)
            - confirmed: Objects that have met the frame threshold
            - pending: Objects still accumulating frames
        """
        self._current_frame_objects.clear()
        
        confirmed = []
        pending = []
        
        # Process each detection
        for det in detections:
            class_name = det['class_name']
            center = det.get('center', self._get_center(det['bbox']))
            bbox = det['bbox']
            confidence = det['confidence']
            
            self._current_frame_objects.add(class_name)
            
            # Check if we're already tracking this object type
            if class_name in self._tracked_objects:
                tracked = self._tracked_objects[class_name]
                
                # Check if detection is close enough to be the same object
                if tracked.last_center:
                    distance = self._calculate_distance(center, tracked.last_center)
                    if distance <= self.proximity_threshold:
                        # Same object, update tracking
                        tracked.update(bbox, confidence, center)
                    else:
                        # Different object of same class, reset tracking
                        tracked.reset()
                        tracked.last_bbox = bbox
                        tracked.last_confidence = confidence
                        tracked.last_center = center
                        tracked.consecutive_count = 1
                else:
                    tracked.update(bbox, confidence, center)
            else:
                # New object to track
                self._tracked_objects[class_name] = TrackedObject(
                    class_name=class_name,
                    consecutive_count=1,
                    last_bbox=bbox,
                    last_confidence=confidence,
                    last_center=center
                )
            
            # Check if object has met threshold
            tracked = self._tracked_objects[class_name]
            
            if tracked.consecutive_count >= self.required_frames:
                confirmed.append({
                    'class_name': class_name,
                    'bbox': tracked.last_bbox,
                    'confidence': tracked.last_confidence,
                    'center': tracked.last_center,
                    'consecutive_frames': tracked.consecutive_count
                })
            else:
                pending.append({
                    'class_name': class_name,
                    'bbox': tracked.last_bbox,
                    'confidence': tracked.last_confidence,
                    'center': tracked.last_center,
                    'consecutive_frames': tracked.consecutive_count
                })
        
        # Reset tracking for objects not seen in this frame
        self._cleanup_missing_objects()
        
        return confirmed, pending
    
    def _cleanup_missing_objects(self) -> None:
        """Reset tracking for objects not detected in current frame."""
        for class_name, tracked in self._tracked_objects.items():
            if class_name not in self._current_frame_objects:
                tracked.reset()
    
    def _calculate_distance(
        self,
        point1: Tuple[int, int],
        point2: Tuple[int, int]
    ) -> float:
        """Calculate Euclidean distance between two points."""
        return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate center of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self._tracked_objects.clear()
        self._current_frame_objects.clear()
    
    def get_tracking_status(self) -> Dict[str, dict]:
        """
        Get current tracking status for all objects.
        
        Returns:
            Dictionary mapping class names to tracking info
        """
        return {
            name: {
                'consecutive_count': obj.consecutive_count,
                'last_confidence': obj.last_confidence,
                'frames_remaining': max(0, self.required_frames - obj.consecutive_count)
            }
            for name, obj in self._tracked_objects.items()
            if obj.consecutive_count > 0
        }
    
    def set_required_frames(self, frames: int) -> None:
        """
        Set the required consecutive frames threshold.
        
        Args:
            frames: Number of frames required
        """
        self.required_frames = max(1, frames)
    
    def set_proximity_threshold(self, threshold: int) -> None:
        """
        Set the proximity threshold for object matching.
        
        Args:
            threshold: Distance in pixels
        """
        self.proximity_threshold = threshold
