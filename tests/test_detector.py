"""
Unit tests for Object Detector Module
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from detection.detector import Detection, ObjectDetector
from detection.preprocessor import FramePreprocessor
from detection.model_loader import ModelLoader


class TestFramePreprocessor:
    """Tests for FramePreprocessor class."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = FramePreprocessor(
            target_size=(640, 480),
            normalize=True
        )
        
        assert preprocessor.target_size == (640, 480)
        assert preprocessor.normalize is True
        assert preprocessor.roi_coordinates is None
    
    def test_resize(self):
        """Test frame resizing."""
        preprocessor = FramePreprocessor(target_size=(320, 240))
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        resized = preprocessor.resize(frame, (320, 240))
        
        assert resized.shape[:2] == (240, 320)
    
    def test_crop_roi(self):
        """Test ROI cropping."""
        preprocessor = FramePreprocessor()
        
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[100:200, 100:200] = 255  # White square
        
        cropped = preprocessor.crop_roi(frame, (100, 100, 200, 200))
        
        assert cropped.shape[:2] == (100, 100)
        assert np.all(cropped > 0)  # All white


class TestDetection:
    """Tests for Detection dataclass."""
    
    def test_detection_creation(self):
        """Test detection object creation."""
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
            area=20000
        )
        
        assert detection.class_name == "person"
        assert detection.confidence == 0.85
        assert detection.bbox == (100, 100, 200, 300)
    
    def test_is_in_roi(self):
        """Test ROI containment check."""
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
            area=20000
        )
        
        # Center is inside
        assert detection.is_in_roi((100, 100, 300, 400)) is True
        
        # Center is outside
        assert detection.is_in_roi((0, 0, 50, 50)) is False
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.85,
            bbox=(100, 100, 200, 300),
            center=(150, 200),
            area=20000
        )
        
        det_dict = detection.to_dict()
        
        assert isinstance(det_dict, dict)
        assert det_dict['class_name'] == "person"
        assert det_dict['confidence'] == 0.85


class TestModelLoader:
    """Tests for ModelLoader class."""
    
    @patch('detection.model_loader.YOLO')
    def test_initialization(self, mock_yolo):
        """Test model loader initialization."""
        loader = ModelLoader(
            model_name="yolov8n.pt",
            use_onnx=False
        )
        
        assert loader.model_name == "yolov8n.pt"
        assert loader.use_onnx is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
