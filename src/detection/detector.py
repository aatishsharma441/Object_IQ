"""
Object Detector Module
Main detection engine using YOLOv8 for real-time object detection.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .model_loader import ModelLoader
from .preprocessor import FramePreprocessor


@dataclass
class Detection:
    """
    Represents a single object detection.
    
    Attributes:
        class_id: COCO class ID
        class_name: Class name (e.g., 'person', 'car')
        confidence: Detection confidence score (0-1)
        bbox: Bounding box coordinates (x1, y1, x2, y2)
        center: Center point of bounding box (cx, cy)
        area: Area of bounding box in pixels
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert detection to dictionary."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "center": self.center,
            "area": self.area
        }
    
    def is_in_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """
        Check if detection center is within a Region of Interest.
        
        Args:
            roi: ROI coordinates (x1, y1, x2, y2)
            
        Returns:
            True if detection center is within ROI
        """
        x1, y1, x2, y2 = roi
        cx, cy = self.center
        return x1 <= cx <= x2 and y1 <= cy <= y2


class ObjectDetector:
    """
    Real-time object detector using YOLOv8.
    Supports filtering by target objects, confidence threshold, and ROI.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        model_dir: Optional[Union[str, Path]] = None,
        use_onnx: bool = False,
        confidence_threshold: float = 0.5,
        target_objects: Optional[List[str]] = None,
        resolution: Tuple[int, int] = (640, 480)
    ):
        """
        Initialize object detector.
        
        Args:
            model_name: YOLOv8 model name
            model_dir: Directory for model storage
            use_onnx: Whether to use ONNX runtime
            confidence_threshold: Minimum confidence for detections
            target_objects: List of object classes to detect (None = all)
            resolution: Input resolution for detection
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.confidence_threshold = confidence_threshold
        self.target_objects = [obj.lower() for obj in target_objects] if target_objects else None
        self.resolution = resolution
        
        # Initialize components
        self.model_loader = ModelLoader(model_name, model_dir, use_onnx)
        self.preprocessor = FramePreprocessor(target_size=resolution)
        
        # Model and class names
        self._model = None
        self._class_names: Dict[int, str] = {}
        
        # Performance tracking
        self._inference_times: List[float] = []
        self._max_time_samples = 100
    
    def initialize(self) -> None:
        """Load the model and initialize detector."""
        self._model = self.model_loader.load()
        self._class_names = self._model.names
    
    def detect(
        self,
        frame: np.ndarray,
        confidence_threshold: Optional[float] = None
    ) -> List[Detection]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame (BGR format)
            confidence_threshold: Override confidence threshold
            
        Returns:
            List of Detection objects
        """
        if self._model is None:
            self.initialize()
        
        conf = confidence_threshold or self.confidence_threshold
        
        # Track inference time
        start_time = time.time()
        
        # Run inference with improved settings for crowd detection
        results = self._model(
            frame, 
            conf=conf, 
            verbose=False,
            iou=0.5,           # IoU threshold for NMS
            max_det=100,       # Maximum detections per frame
            agnostic_nms=True  # Class-agnostic NMS for better crowd detection
        )
        
        # Track inference time
        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)
        if len(self._inference_times) > self._max_time_samples:
            self._inference_times.pop(0)
        
        # Parse results
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for i in range(len(boxes)):
                # Get box coordinates
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Get class and confidence
                class_id = int(boxes.cls[i].cpu().numpy())
                confidence = float(boxes.conf[i].cpu().numpy())
                class_name = self._class_names.get(class_id, f"class_{class_id}")
                
                # Filter by target objects if specified
                if self.target_objects and class_name.lower() not in self.target_objects:
                    continue
                
                # Calculate center and area
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                area = (x2 - x1) * (y2 - y1)
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=center,
                    area=area
                )
                detections.append(detection)
        
        return detections
    
    def detect_with_filter(
        self,
        frame: np.ndarray,
        target_objects: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> List[Detection]:
        """
        Detect objects with additional filtering options.
        
        Args:
            frame: Input frame
            target_objects: Filter by object types
            min_confidence: Minimum confidence
            roi: Filter by Region of Interest
            
        Returns:
            Filtered list of Detection objects
        """
        detections = self.detect(frame, min_confidence)
        
        # Filter by target objects
        if target_objects:
            target_lower = [t.lower() for t in target_objects]
            detections = [d for d in detections if d.class_name.lower() in target_lower]
        
        # Filter by ROI
        if roi:
            detections = [d for d in detections if d.is_in_roi(roi)]
        
        return detections
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True,
        draw_confidence: bool = True,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: Input frame
            detections: List of detections to draw
            draw_labels: Whether to draw class labels
            draw_confidence: Whether to draw confidence scores
            color: Bounding box color (None = class-based colors)
            thickness: Line thickness
            
        Returns:
            Frame with detections drawn
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Get color based on class if not specified
            box_color = color or self._get_class_color(det.class_id)
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), box_color, thickness)
            
            # Draw label
            if draw_labels:
                label = det.class_name
                if draw_confidence:
                    label += f" {det.confidence:.0%}"
                
                # Calculate label position and size
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    output,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    box_color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    output,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        
        return output
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """
        Generate a consistent color for a class ID.
        
        Args:
            class_id: Class ID
            
        Returns:
            BGR color tuple
        """
        # Predefined colors for common classes
        color_map = {
            0: (255, 0, 0),      # Person - Blue
            1: (0, 255, 0),      # Bicycle - Green
            2: (0, 0, 255),      # Car - Red
            3: (255, 255, 0),    # Motorcycle - Cyan
            5: (255, 0, 255),    # Bus - Magenta
            7: (0, 255, 255),    # Truck - Yellow
            15: (128, 0, 128),   # Cat - Purple
            16: (128, 128, 0),   # Dog - Olive
        }
        
        if class_id in color_map:
            return color_map[class_id]
        
        # Generate color based on class ID
        np.random.seed(class_id * 123)
        return tuple(map(int, np.random.randint(50, 255, 3)))
    
    def get_avg_inference_time(self) -> float:
        """
        Get average inference time over recent frames.
        
        Returns:
            Average inference time in seconds
        """
        if not self._inference_times:
            return 0.0
        return sum(self._inference_times) / len(self._inference_times)
    
    def get_fps(self) -> float:
        """
        Get estimated FPS based on inference time.
        
        Returns:
            Estimated FPS
        """
        avg_time = self.get_avg_inference_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return self.model_loader.get_model_info()
    
    def set_target_objects(self, objects: List[str]) -> None:
        """
        Set target objects for detection filtering.
        
        Args:
            objects: List of object class names
        """
        self.target_objects = [obj.lower() for obj in objects]
    
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Set confidence threshold for detections.
        
        Args:
            threshold: Minimum confidence (0-1)
        """
        self.confidence_threshold = threshold
    
    def set_resolution(self, resolution: Tuple[int, int]) -> None:
        """
        Set detection resolution.
        
        Args:
            resolution: (width, height) tuple
        """
        self.resolution = resolution
        self.preprocessor.set_target_size(resolution)
