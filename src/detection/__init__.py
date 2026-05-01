"""
Object Detection Module
Contains YOLOv8 detection engine, DeepSORT tracking, weapon detection,
and crowd heatmap visualization utilities.
"""

from .detector import ObjectDetector, Detection
from .preprocessor import FramePreprocessor
from .model_loader import ModelLoader
from .tracker import DeepSORTTracker, Track, CrowdTracker
from .weapon_detector import WeaponDetector, WeaponClassifier, WeaponDetection
from .heatmap import CrowdHeatmap, CrowdAnalytics, HeatmapConfig

__all__ = [
    "ObjectDetector",
    "Detection",
    "FramePreprocessor",
    "ModelLoader",
    "DeepSORTTracker",
    "Track",
    "CrowdTracker",
    "WeaponDetector",
    "WeaponClassifier",
    "WeaponDetection",
    "CrowdHeatmap",
    "CrowdAnalytics",
    "HeatmapConfig"
]
