"""
Unit tests for Logic Engine Module
"""

import sys
import time
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from logic.temporal_filter import TemporalFilter, TrackedObject
from logic.cooldown_manager import CooldownManager
from logic.roi_manager import ROIManager, ROIZone
from logic.decision_engine import DecisionEngine, AlertDecision


class TestTemporalFilter:
    """Tests for TemporalFilter class."""
    
    def test_initialization(self):
        """Test temporal filter initialization."""
        tf = TemporalFilter(required_frames=3)
        
        assert tf.required_frames == 3
    
    def test_single_frame_detection(self):
        """Test that single frame detection doesn't trigger alert."""
        tf = TemporalFilter(required_frames=3)
        
        detections = [{
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': (100, 100, 200, 300),
            'center': (150, 200)
        }]
        
        confirmed, pending = tf.process_frame(detections)
        
        assert len(confirmed) == 0
        assert len(pending) == 1
    
    def test_consecutive_frame_detection(self):
        """Test that consecutive frames trigger alert."""
        tf = TemporalFilter(required_frames=2)
        
        detections = [{
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': (100, 100, 200, 300),
            'center': (150, 200)
        }]
        
        # First frame
        confirmed1, pending1 = tf.process_frame(detections)
        assert len(confirmed1) == 0
        
        # Second frame
        confirmed2, pending2 = tf.process_frame(detections)
        assert len(confirmed2) == 1
    
    def test_reset_on_missing_detection(self):
        """Test that counter resets when object not detected."""
        tf = TemporalFilter(required_frames=3)
        
        detections = [{
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': (100, 100, 200, 300),
            'center': (150, 200)
        }]
        
        # First frame
        tf.process_frame(detections)
        
        # Empty frame (no detections)
        tf.process_frame([])
        
        # Detection again
        confirmed, pending = tf.process_frame(detections)
        
        # Should start from 1 again
        assert len(confirmed) == 0
    
    def test_reset(self):
        """Test filter reset."""
        tf = TemporalFilter(required_frames=3)
        
        detections = [{
            'class_name': 'person',
            'confidence': 0.85,
            'bbox': (100, 100, 200, 300),
            'center': (150, 200)
        }]
        
        tf.process_frame(detections)
        tf.reset()
        
        status = tf.get_tracking_status()
        assert len(status) == 0


class TestCooldownManager:
    """Tests for CooldownManager class."""
    
    def test_initialization(self):
        """Test cooldown manager initialization."""
        cm = CooldownManager(default_cooldown_seconds=60)
        
        assert cm.default_cooldown_seconds == 60
    
    def test_not_on_cooldown_initially(self):
        """Test that object is not on cooldown initially."""
        cm = CooldownManager()
        
        assert cm.is_on_cooldown('person') is False
    
    def test_start_and_check_cooldown(self):
        """Test starting and checking cooldown."""
        cm = CooldownManager(default_cooldown_seconds=60)
        
        cm.start_cooldown('person')
        
        assert cm.is_on_cooldown('person') is True
    
    def test_cooldown_override(self):
        """Test object-specific cooldown override."""
        cm = CooldownManager(default_cooldown_seconds=60)
        cm.set_cooldown_override('person', 30)
        
        assert cm.get_cooldown_duration('person') == 30
        assert cm.get_cooldown_duration('car') == 60
    
    def test_reset_cooldown(self):
        """Test resetting cooldown."""
        cm = CooldownManager()
        
        cm.start_cooldown('person')
        assert cm.is_on_cooldown('person') is True
        
        cm.reset_cooldown('person')
        assert cm.is_on_cooldown('person') is False
    
    def test_remaining_cooldown(self):
        """Test remaining cooldown calculation."""
        cm = CooldownManager(default_cooldown_seconds=10)
        
        cm.start_cooldown('person')
        remaining = cm.get_remaining_cooldown('person')
        
        assert remaining > 0
        assert remaining <= 10


class TestROIManager:
    """Tests for ROIManager class."""
    
    def test_initialization(self):
        """Test ROI manager initialization."""
        roi = ROIManager()
        
        assert roi.get_zone_count() == 0
    
    def test_add_zone(self):
        """Test adding ROI zone."""
        roi = ROIManager()
        roi.add_zone('entrance', (0, 0, 100, 100))
        
        assert roi.get_zone_count() == 1
    
    def test_point_in_zone(self):
        """Test point containment in zone."""
        roi = ROIManager()
        roi.add_zone('entrance', (0, 0, 100, 100))
        
        in_zone, zone_name = roi.is_in_any_zone((50, 50))
        
        assert in_zone is True
        assert zone_name == 'entrance'
    
    def test_point_outside_zone(self):
        """Test point outside zone."""
        roi = ROIManager()
        roi.add_zone('entrance', (0, 0, 100, 100))
        
        in_zone, zone_name = roi.is_in_any_zone((200, 200))
        
        assert in_zone is False
        assert zone_name is None
    
    def test_remove_zone(self):
        """Test removing zone."""
        roi = ROIManager()
        roi.add_zone('entrance', (0, 0, 100, 100))
        
        result = roi.remove_zone('entrance')
        
        assert result is True
        assert roi.get_zone_count() == 0
    
    def test_filter_detections(self):
        """Test filtering detections by ROI."""
        roi = ROIManager()
        roi.add_zone('entrance', (0, 0, 100, 100))
        
        detections = [
            {'bbox': (10, 10, 50, 50), 'class_name': 'person'},  # Inside
            {'bbox': (200, 200, 300, 300), 'class_name': 'car'}  # Outside
        ]
        
        filtered = roi.filter_detections(detections)
        
        assert len(filtered) == 1
        assert filtered[0]['class_name'] == 'person'


class TestDecisionEngine:
    """Tests for DecisionEngine class."""
    
    def test_initialization(self):
        """Test decision engine initialization."""
        engine = DecisionEngine(
            target_objects=['person', 'car'],
            confidence_threshold=0.7,
            consecutive_frames=3,
            cooldown_seconds=60
        )
        
        assert engine.default_confidence_threshold == 0.7
        assert engine.default_consecutive_frames == 3
    
    def test_filter_target_objects(self):
        """Test filtering by target objects."""
        engine = DecisionEngine(
            target_objects=['person'],
            confidence_threshold=0.5,
            consecutive_frames=1,
            cooldown_seconds=60
        )
        
        detections = [
            {'class_name': 'person', 'confidence': 0.8, 'bbox': (0, 0, 100, 100), 'center': (50, 50)},
            {'class_name': 'car', 'confidence': 0.8, 'bbox': (0, 0, 100, 100), 'center': (50, 50)}
        ]
        
        decisions = engine.process_detections(detections)
        
        # Only person should be processed
        assert all(d.object_type == 'person' for d in decisions)
    
    def test_filter_confidence(self):
        """Test filtering by confidence threshold."""
        engine = DecisionEngine(
            target_objects=['person'],
            confidence_threshold=0.7,
            consecutive_frames=1,
            cooldown_seconds=60
        )
        
        detections = [
            {'class_name': 'person', 'confidence': 0.8, 'bbox': (0, 0, 100, 100), 'center': (50, 50)},
            {'class_name': 'person', 'confidence': 0.5, 'bbox': (0, 0, 100, 100), 'center': (50, 50)}
        ]
        
        decisions = engine.process_detections(detections)
        
        # Only high confidence should pass
        assert all(d.confidence >= 0.7 for d in decisions if d.consecutive_frames > 0)
    
    def test_statistics(self):
        """Test statistics retrieval."""
        engine = DecisionEngine(
            target_objects=['person'],
            confidence_threshold=0.5,
            consecutive_frames=1,
            cooldown_seconds=60
        )
        
        stats = engine.get_statistics()
        
        assert 'total_detections' in stats
        assert 'total_alerts' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
