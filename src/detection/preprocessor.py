"""
Frame Preprocessor Module
Handles preprocessing of video frames before detection.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


class FramePreprocessor:
    """
    Preprocesses video frames for object detection.
    Handles resizing, normalization, and ROI cropping.
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 480),
        normalize: bool = True,
        roi_coordinates: Optional[Tuple[int, int, int, int]] = None
    ):
        """
        Initialize frame preprocessor.
        
        Args:
            target_size: Target frame size (width, height)
            normalize: Whether to normalize pixel values
            roi_coordinates: ROI coordinates (x1, y1, x2, y2) for cropping
        """
        self.target_size = target_size
        self.normalize = normalize
        self.roi_coordinates = roi_coordinates
    
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a frame for detection.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Preprocessed frame
        """
        # Apply ROI cropping if specified
        if self.roi_coordinates:
            frame = self.crop_roi(frame, self.roi_coordinates)
        
        # Resize frame
        frame = self.resize(frame, self.target_size)
        
        return frame
    
    def resize(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int],
        keep_aspect_ratio: bool = False
    ) -> np.ndarray:
        """
        Resize frame to target size.
        
        Args:
            frame: Input frame
            target_size: Target size (width, height)
            keep_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized frame
        """
        if keep_aspect_ratio:
            # Calculate scaling factor
            h, w = frame.shape[:2]
            target_w, target_h = target_size
            
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # Resize with aspect ratio
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad if necessary
            if new_w != target_w or new_h != target_h:
                pad_w = target_w - new_w
                pad_h = target_h - new_h
                top = pad_h // 2
                bottom = pad_h - top
                left = pad_w // 2
                right = pad_w - left
                
                resized = cv2.copyMakeBorder(
                    resized, top, bottom, left, right,
                    cv2.BORDER_CONSTANT, value=(114, 114, 114)
                )
            
            return resized
        else:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    
    def crop_roi(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Crop frame to Region of Interest.
        
        Args:
            frame: Input frame
            roi: ROI coordinates (x1, y1, x2, y2)
            
        Returns:
            Cropped frame
        """
        x1, y1, x2, y2 = roi
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        return frame[y1:y2, x1:x2]
    
    def normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            frame: Input frame (0-255 range)
            
        Returns:
            Normalized frame
        """
        return frame.astype(np.float32) / 255.0
    
    def convert_color(
        self,
        frame: np.ndarray,
        color_space: str = "RGB"
    ) -> np.ndarray:
        """
        Convert frame color space.
        
        Args:
            frame: Input frame (BGR)
            color_space: Target color space (RGB, GRAY, HSV)
            
        Returns:
            Converted frame
        """
        if color_space == "RGB":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        elif color_space == "GRAY":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif color_space == "HSV":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        else:
            return frame
    
    def draw_roi_overlay(
        self,
        frame: np.ndarray,
        roi: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw ROI rectangle on frame.
        
        Args:
            frame: Input frame
            roi: ROI coordinates (x1, y1, x2, y2)
            color: Rectangle color (BGR)
            thickness: Line thickness
            
        Returns:
            Frame with ROI overlay
        """
        x1, y1, x2, y2 = roi
        return cv2.rectangle(frame.copy(), (x1, y1), (x2, y2), color, thickness)
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> None:
        """
        Set ROI coordinates for preprocessing.
        
        Args:
            roi: ROI coordinates (x1, y1, x2, y2)
        """
        self.roi_coordinates = roi
    
    def set_target_size(self, size: Tuple[int, int]) -> None:
        """
        Set target size for preprocessing.
        
        Args:
            size: Target size (width, height)
        """
        self.target_size = size
