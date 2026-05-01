"""
Weapon Detection Module
Specialized detection for weapons (guns, knives) with improved classification.
Fixes the issue where guns are misclassified as cell phones.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import os


@dataclass
class WeaponDetection:
    """Represents a weapon detection result."""
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    is_weapon: bool
    weapon_type: str
    threat_level: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox,
            'is_weapon': self.is_weapon,
            'weapon_type': self.weapon_type,
            'threat_level': self.threat_level
        }


class WeaponClassifier:
    """
    Specialized weapon classifier to improve weapon detection.
    Uses shape analysis and feature extraction to distinguish weapons
    from common misclassifications like cell phones.
    """
    
    # Known weapon classes
    WEAPON_CLASSES = ['gun', 'knife', 'pistol', 'rifle', 'shotgun', 'weapon', 'firearm']
    
    # Classes commonly confused with weapons
    CONFUSION_CLASSES = ['cell phone', 'remote', 'mouse', 'scissors', 'toothbrush', 'fork', 'spoon']
    
    # Aspect ratio ranges for different weapons
    WEAPON_ASPECT_RATIOS = {
        'gun': (1.2, 4.0),      # Guns are typically wider than tall
        'pistol': (1.3, 3.0),   # Handguns
        'rifle': (2.0, 6.0),    # Long guns
        'knife': (0.3, 1.5),    # Knives can vary but often taller than wide
        'dagger': (0.2, 1.0),   # Daggers are taller than wide
    }
    
    def __init__(self):
        """Initialize weapon classifier."""
        self._load_weapon_patterns()
        
    def _load_weapon_patterns(self):
        """Load weapon detection patterns."""
        # Color features for weapon detection
        self.weapon_color_ranges = {
            'metallic': {
                'low': np.array([0, 0, 100]),
                'high': np.array([180, 50, 220])
            },
            'dark': {
                'low': np.array([0, 0, 0]),
                'high': np.array([180, 255, 80])
            }
        }
    
    def analyze_shape(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze shape features of detected region.
        
        Args:
            bbox: Bounding box coordinates
            frame: Input frame
            
        Returns:
            Dictionary of shape features
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'aspect_ratio': 1.0, 'solidity': 0.0, 'extent': 0.0}
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {'aspect_ratio': 1.0, 'solidity': 0.0, 'extent': 0.0}
        
        # Calculate features
        width = x2 - x1
        height = y2 - y1
        
        aspect_ratio = width / max(height, 1)
        
        # Convert to grayscale and threshold
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        solidity = 0.0
        extent = 0.0
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            
            # Bounding rectangle area
            rect_area = width * height
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            
            # Solidity (contour area / hull area)
            solidity = contour_area / max(hull_area, 1)
            
            # Extent (contour area / bounding rectangle area)
            extent = contour_area / max(rect_area, 1)
        
        return {
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'extent': extent
        }
    
    def analyze_color(self, bbox: Tuple[int, int, int, int], frame: np.ndarray) -> Dict[str, float]:
        """
        Analyze color features for weapon detection.
        
        Args:
            bbox: Bounding box coordinates
            frame: Input frame
            
        Returns:
            Dictionary of color features
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure valid bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return {'metallic_ratio': 0.0, 'dark_ratio': 0.0}
        
        # Extract ROI
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {'metallic_ratio': 0.0, 'dark_ratio': 0.0}
        
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color ratios
        metallic_mask = cv2.inRange(
            hsv,
            self.weapon_color_ranges['metallic']['low'],
            self.weapon_color_ranges['metallic']['high']
        )
        
        dark_mask = cv2.inRange(
            hsv,
            self.weapon_color_ranges['dark']['low'],
            self.weapon_color_ranges['dark']['high']
        )
        
        total_pixels = roi.shape[0] * roi.shape[1]
        
        metallic_ratio = cv2.countNonZero(metallic_mask) / max(total_pixels, 1)
        dark_ratio = cv2.countNonZero(dark_mask) / max(total_pixels, 1)
        
        return {
            'metallic_ratio': metallic_ratio,
            'dark_ratio': dark_ratio
        }
    
    def classify_detection(
        self,
        detection: Any,
        frame: np.ndarray,
        original_class: str
    ) -> WeaponDetection:
        """
        Classify detection as weapon or not.
        
        Args:
            detection: Detection object with bbox
            frame: Input frame
            original_class: Original class name from YOLO
            
        Returns:
            WeaponDetection object
        """
        bbox = detection.bbox
        confidence = detection.confidence
        
        # Analyze shape
        shape_features = self.analyze_shape(bbox, frame)
        
        # Analyze color
        color_features = self.analyze_color(bbox, frame)
        
        # Determine if this is likely a weapon
        is_weapon = False
        weapon_type = 'unknown'
        threat_level = 'low'
        
        aspect_ratio = shape_features['aspect_ratio']
        metallic_ratio = color_features['metallic_ratio']
        dark_ratio = color_features['dark_ratio']
        
        # Weapon detection logic
        
        # Check for gun-like features
        if original_class.lower() in self.CONFUSION_CLASSES:
            # Check if it might actually be a gun
            if (self.WEAPON_ASPECT_RATIOS['gun'][0] <= aspect_ratio <= self.WEAPON_ASPECT_RATIOS['gun'][1]):
                if metallic_ratio > 0.1 or dark_ratio > 0.3:
                    is_weapon = True
                    weapon_type = 'gun'
                    threat_level = 'high'
                    confidence = min(confidence + 0.15, 0.95)
        
        # Check if original class is already a weapon
        if original_class.lower() in self.WEAPON_CLASSES:
            is_weapon = True
            weapon_type = original_class.lower()
            threat_level = 'high'
        
        # Additional gun detection from shape
        if aspect_ratio >= 1.5 and aspect_ratio <= 4.0:
            if metallic_ratio > 0.15 or dark_ratio > 0.4:
                if original_class.lower() in self.CONFUSION_CLASSES:
                    is_weapon = True
                    weapon_type = 'gun'
                    threat_level = 'high'
        
        # Knife detection
        if aspect_ratio <= 1.5 and aspect_ratio >= 0.3:
            if metallic_ratio > 0.1:
                if original_class.lower() in ['scissors', 'fork', 'spoon', 'knife']:
                    is_weapon = True
                    weapon_type = 'knife'
                    threat_level = 'medium'
        
        return WeaponDetection(
            class_name=weapon_type if is_weapon else original_class,
            confidence=confidence,
            bbox=bbox,
            is_weapon=is_weapon,
            weapon_type=weapon_type if is_weapon else None,
            threat_level=threat_level
        )


class WeaponDetector:
    """
    Main weapon detection class that integrates with the object detector.
    """
    
    # Standard COCO classes that might be weapons
    SUSPICIOUS_COCO_CLASSES = [
        'scissors',        # Can be a weapon
        'knife',           # COCO doesn't have this, but we handle it
        'fork',            # Can be mistaken for knife
    ]
    
    # Our custom weapon classes
    WEAPON_CLASSES = ['gun', 'knife', 'pistol', 'rifle', 'shotgun', 'weapon', 'firearm', 'blade', 'dagger']
    
    def __init__(self, confidence_threshold: float = 0.5):
        """
        Initialize weapon detector.
        
        Args:
            confidence_threshold: Minimum confidence for weapon detection
        """
        self.confidence_threshold = confidence_threshold
        self.classifier = WeaponClassifier()
        
        # Track recent detections for temporal smoothing
        self._recent_detections = []
        self._max_history = 10
    
    def filter_detections(
        self, 
        detections: List[Any], 
        frame: np.ndarray
    ) -> Tuple[List[Any], List[WeaponDetection]]:
        """
        Filter detections and identify weapons.
        
        Args:
            detections: List of Detection objects
            frame: Input frame
            
        Returns:
            Tuple of (normal_detections, weapon_detections)
        """
        normal_detections = []
        weapon_detections = []
        
        for det in detections:
            class_name = det.class_name.lower()
            
            # Check if this is potentially a weapon
            if class_name in self.WEAPON_CLASSES:
                # Classify for confirmation
                weapon_det = self.classifier.classify_detection(det, frame, class_name)
                weapon_detections.append(weapon_det)
                
            elif class_name in self.classifier.CONFUSION_CLASSES:
                # Check if misclassified weapon
                weapon_det = self.classifier.classify_detection(det, frame, class_name)
                
                if weapon_det.is_weapon:
                    weapon_detections.append(weapon_det)
                else:
                    normal_detections.append(det)
                    
            elif class_name in self.SUSPICIOUS_COCO_CLASSES:
                # Scissors and similar objects - check for weapon potential
                weapon_det = self.classifier.classify_detection(det, frame, class_name)
                
                if weapon_det.is_weapon:
                    weapon_detections.append(weapon_det)
                else:
                    normal_detections.append(det)
            else:
                normal_detections.append(det)
        
        return normal_detections, weapon_detections
    
    def get_weapon_color(self, threat_level: str) -> Tuple[int, int, int]:
        """Get color for drawing weapon detection based on threat level."""
        colors = {
            'low': (0, 255, 255),      # Yellow
            'medium': (0, 165, 255),   # Orange
            'high': (0, 0, 255),       # Red
        }
        return colors.get(threat_level, (0, 0, 255))
    
    def draw_weapon_detections(
        self,
        frame: np.ndarray,
        weapon_detections: List[WeaponDetection]
    ) -> np.ndarray:
        """
        Draw weapon detections on frame with special styling.
        
        Args:
            frame: Input frame
            weapon_detections: List of WeaponDetection objects
            
        Returns:
            Frame with weapon detections drawn
        """
        output = frame.copy()
        
        for wdet in weapon_detections:
            if not wdet.is_weapon:
                continue
                
            x1, y1, x2, y2 = wdet.bbox
            color = self.get_weapon_color(wdet.threat_level)
            
            # Draw thick bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            
            # Draw warning triangle
            triangle_pts = np.array([
                [x2 - 30, y1],
                [x2, y1],
                [x2 - 15, y1 + 20]
            ])
            cv2.fillPoly(output, [triangle_pts], color)
            
            # Draw exclamation mark
            cv2.putText(output, "!", (x2 - 22, y1 + 17),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw label
            label = f"WEAPON: {wdet.weapon_type.upper()}"
            cv2.rectangle(output, (x1, y2), (x1 + len(label) * 10 + 20, y2 + 25), color, -1)
            cv2.putText(output, label, (x1 + 5, y2 + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw threat level
            threat_label = f"THREAT: {wdet.threat_level.upper()}"
            cv2.putText(output, threat_label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return output


def create_weapon_detector(confidence_threshold: float = 0.5) -> WeaponDetector:
    """Factory function to create weapon detector."""
    return WeaponDetector(confidence_threshold=confidence_threshold)
