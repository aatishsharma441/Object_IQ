"""
Decision Engine Module
Core intelligence layer that combines all filtering and decision logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .temporal_filter import TemporalFilter
from .cooldown_manager import CooldownManager
from .roi_manager import ROIManager


@dataclass
class AlertDecision:
    """
    Represents an alert decision from the engine.
    
    Attributes:
        should_alert: Whether an alert should be sent
        object_type: Type of detected object
        confidence: Detection confidence
        bbox: Bounding box coordinates
        consecutive_frames: Number of consecutive frames detected
        roi_zone: ROI zone name (if applicable)
        reason: Reason for the decision
        timestamp: Decision timestamp
        metadata: Additional metadata
    """
    should_alert: bool
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    consecutive_frames: int = 1
    roi_zone: Optional[str] = None
    reason: str = ""
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'should_alert': self.should_alert,
            'object_type': self.object_type,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'consecutive_frames': self.consecutive_frames,
            'roi_zone': self.roi_zone,
            'reason': self.reason,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata
        }


class DecisionEngine:
    """
    Core intelligence layer for detection decisions.
    Combines temporal filtering, cooldown management, and ROI filtering.
    """
    
    def __init__(
        self,
        target_objects: List[str],
        confidence_threshold: float = 0.70,
        consecutive_frames: int = 3,
        cooldown_seconds: int = 60,
        roi_enabled: bool = False,
        roi_zones: Optional[List[Dict]] = None,
        object_config: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize decision engine.
        
        Args:
            target_objects: List of object types to detect
            confidence_threshold: Default minimum confidence
            consecutive_frames: Required consecutive frames for alert
            cooldown_seconds: Default cooldown between alerts
            roi_enabled: Whether ROI filtering is enabled
            roi_zones: List of ROI zone configurations
            object_config: Object-specific configurations
        """
        self.target_objects = [obj.lower() for obj in target_objects]
        self.default_confidence_threshold = confidence_threshold
        self.default_consecutive_frames = consecutive_frames
        self.default_cooldown_seconds = cooldown_seconds
        
        # Object-specific configuration
        self.object_config = object_config or {}
        
        # Initialize sub-components
        self.temporal_filter = TemporalFilter(required_frames=consecutive_frames)
        self.cooldown_manager = CooldownManager(default_cooldown_seconds=cooldown_seconds)
        self.roi_manager = ROIManager(zones=roi_zones if roi_enabled else None)
        
        # Apply object-specific cooldown overrides
        for obj_type, config in self.object_config.items():
            if 'cooldown_override' in config:
                self.cooldown_manager.set_cooldown_override(
                    obj_type, config['cooldown_override']
                )
        
        # ROI enabled state
        self.roi_enabled = roi_enabled
        
        # Statistics
        self._total_detections = 0
        self._total_alerts = 0
        self._rejected_by_cooldown = 0
        self._rejected_by_confidence = 0
        self._rejected_by_roi = 0
    
    def process_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[AlertDecision]:
        """
        Process a list of detections and make alert decisions.
        
        Args:
            detections: List of detection dictionaries with:
                       - class_name: str
                       - confidence: float
                       - bbox: Tuple[int, int, int, int]
                       - center: Tuple[int, int]
            
        Returns:
            List of AlertDecision objects
        """
        decisions = []
        
        # Step 1: Filter by target objects
        filtered = self._filter_target_objects(detections)
        
        # Step 2: Filter by confidence threshold (per object)
        filtered = self._filter_confidence(filtered)
        
        # Step 3: Apply temporal filtering
        confirmed, pending = self.temporal_filter.process_frame(filtered)
        
        # Process confirmed detections (met temporal threshold)
        for det in confirmed:
            decision = self._make_alert_decision(det, confirmed=True)
            decisions.append(decision)
        
        # Track pending detections for logging
        for det in pending:
            decisions.append(AlertDecision(
                should_alert=False,
                object_type=det['class_name'],
                confidence=det['confidence'],
                bbox=det['bbox'],
                consecutive_frames=det.get('consecutive_frames', 1),
                reason=f"Temporal filter pending ({det.get('consecutive_frames', 1)}/{self._get_required_frames(det['class_name'])} frames)"
            ))
        
        return decisions
    
    def _filter_target_objects(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter detections to only include target objects."""
        return [
            det for det in detections
            if det.get('class_name', '').lower() in self.target_objects
        ]
    
    def _filter_confidence(
        self,
        detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter detections by confidence threshold."""
        filtered = []
        
        for det in detections:
            obj_type = det.get('class_name', '').lower()
            threshold = self._get_confidence_threshold(obj_type)
            
            if det.get('confidence', 0) >= threshold:
                filtered.append(det)
            else:
                self._rejected_by_confidence += 1
        
        return filtered
    
    def _make_alert_decision(
        self,
        detection: Dict[str, Any],
        confirmed: bool
    ) -> AlertDecision:
        """
        Make an alert decision for a confirmed detection.
        
        Args:
            detection: Detection dictionary
            confirmed: Whether temporal threshold was met
            
        Returns:
            AlertDecision object
        """
        obj_type = detection.get('class_name', 'unknown')
        confidence = detection.get('confidence', 0)
        bbox = detection.get('bbox', (0, 0, 0, 0))
        consecutive_frames = detection.get('consecutive_frames', 1)
        
        self._total_detections += 1
        
        # Check ROI if enabled
        roi_zone = None
        if self.roi_enabled:
            in_zone, zone_name = self.roi_manager.is_bbox_in_any_zone(bbox)
            if not in_zone:
                self._rejected_by_roi += 1
                return AlertDecision(
                    should_alert=False,
                    object_type=obj_type,
                    confidence=confidence,
                    bbox=bbox,
                    consecutive_frames=consecutive_frames,
                    reason="Outside ROI zones"
                )
            roi_zone = zone_name
        
        # Check cooldown
        if self.cooldown_manager.is_on_cooldown(obj_type):
            self._rejected_by_cooldown += 1
            remaining = self.cooldown_manager.get_remaining_cooldown(obj_type)
            return AlertDecision(
                should_alert=False,
                object_type=obj_type,
                confidence=confidence,
                bbox=bbox,
                consecutive_frames=consecutive_frames,
                roi_zone=roi_zone,
                reason=f"On cooldown ({remaining}s remaining)"
            )
        
        # All checks passed - trigger alert
        self._total_alerts += 1
        
        # Start cooldown for this object type
        self.cooldown_manager.start_cooldown(obj_type)
        
        return AlertDecision(
            should_alert=True,
            object_type=obj_type,
            confidence=confidence,
            bbox=bbox,
            consecutive_frames=consecutive_frames,
            roi_zone=roi_zone,
            reason="All conditions satisfied"
        )
    
    def _get_confidence_threshold(self, object_type: str) -> float:
        """Get confidence threshold for an object type."""
        obj_config = self.object_config.get(object_type.lower(), {})
        return obj_config.get('confidence_override', self.default_confidence_threshold)
    
    def _get_required_frames(self, object_type: str) -> int:
        """Get required consecutive frames for an object type."""
        # Could be extended for object-specific requirements
        return self.default_consecutive_frames
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get decision engine statistics.
        
        Returns:
            Statistics dictionary
        """
        return {
            'total_detections': self._total_detections,
            'total_alerts': self._total_alerts,
            'rejected_by_cooldown': self._rejected_by_cooldown,
            'rejected_by_confidence': self._rejected_by_confidence,
            'rejected_by_roi': self._rejected_by_roi,
            'objects_on_cooldown': self.cooldown_manager.get_objects_on_cooldown(),
            'tracking_status': self.temporal_filter.get_tracking_status()
        }
    
    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._total_detections = 0
        self._total_alerts = 0
        self._rejected_by_cooldown = 0
        self._rejected_by_confidence = 0
        self._rejected_by_roi = 0
    
    def reset_all(self) -> None:
        """Reset all state (temporal filter, cooldowns, statistics)."""
        self.temporal_filter.reset()
        self.cooldown_manager.reset_all()
        self.reset_statistics()
    
    def set_roi_enabled(self, enabled: bool) -> None:
        """Enable or disable ROI filtering."""
        self.roi_enabled = enabled
    
    def add_roi_zone(
        self,
        name: str,
        coordinates: Tuple[int, int, int, int]
    ) -> None:
        """Add an ROI zone."""
        self.roi_manager.add_zone(name, coordinates)
    
    def remove_roi_zone(self, name: str) -> bool:
        """Remove an ROI zone."""
        return self.roi_manager.remove_zone(name)
    
    def update_object_config(
        self,
        object_type: str,
        config: Dict[str, Any]
    ) -> None:
        """
        Update configuration for a specific object type.
        
        Args:
            object_type: Object class name
            config: Configuration dictionary
        """
        self.object_config[object_type.lower()] = config
        
        if 'cooldown_override' in config:
            self.cooldown_manager.set_cooldown_override(
                object_type, config['cooldown_override']
            )
