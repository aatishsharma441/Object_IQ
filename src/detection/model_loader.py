"""
Model Loader Module
Handles downloading, loading, and optimizing YOLOv8 models.
"""

import os
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO


class ModelLoader:
    """
    Handles loading and optimization of YOLOv8 models.
    Supports both PyTorch and ONNX formats for optimal performance.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        model_dir: Optional[Union[str, Path]] = None,
        use_onnx: bool = False
    ):
        """
        Initialize model loader.
        
        Args:
            model_name: Name of the YOLOv8 model (e.g., yolov8n.pt, yolov8s.pt)
            model_dir: Directory to store/load models
            use_onnx: Whether to use ONNX runtime for inference
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        
        # Set model directory
        if model_dir:
            self.model_dir = Path(model_dir)
        else:
            self.model_dir = Path(__file__).parent.parent.parent / "models"
        
        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model instance
        self._model: Optional[YOLO] = None
    
    @property
    def model_path(self) -> Path:
        """Get the path to the model file."""
        if os.path.isabs(self.model_name):
            return Path(self.model_name)
        return self.model_dir / self.model_name
    
    @property
    def onnx_path(self) -> Path:
        """Get the path to the ONNX model file."""
        return self.model_path.with_suffix('.onnx')
    
    def load(self) -> YOLO:
        """
        Load the YOLOv8 model.
        
        Returns:
            Loaded YOLO model instance
        """
        model_path = self.model_path
        
        # Check if model exists locally
        if not model_path.exists():
            # YOLO will auto-download if model name is valid
            # The model will be downloaded to the current directory or cache
            self._model = YOLO(self.model_name)
        else:
            self._model = YOLO(str(model_path))
        
        # Export to ONNX if needed and not exists
        if self.use_onnx and not self.onnx_path.exists():
            self.export_onnx()
        
        # Load ONNX model if specified
        if self.use_onnx:
            self._model = YOLO(str(self.onnx_path))
        
        return self._model
    
    def export_onnx(self, simplify: bool = True) -> Path:
        """
        Export the model to ONNX format for better ARM performance.
        
        Args:
            simplify: Whether to simplify the ONNX model
            
        Returns:
            Path to the exported ONNX model
        """
        if self._model is None:
            self.load()
        
        onnx_path = self._model.export(
            format='onnx',
            simplify=simplify,
            opset=12,
            dynamic=False
        )
        
        return Path(onnx_path)
    
    def get_class_names(self) -> dict:
        """
        Get the class names from the model.
        
        Returns:
            Dictionary mapping class indices to names
        """
        if self._model is None:
            self.load()
        
        return self._model.names
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self._model is None:
            self.load()
        
        return {
            "model_name": self.model_name,
            "model_type": "ONNX" if self.use_onnx else "PyTorch",
            "model_path": str(self.model_path),
            "num_classes": len(self._model.names),
            "class_names": list(self._model.names.values())[:10]  
        }


def load_model(
    model_name: str = "yolov8n.pt",
    model_dir: Optional[Union[str, Path]] = None,
    use_onnx: bool = False
) -> YOLO:
    """
    Convenience function to load a YOLOv8 model.
    
    Args:
        model_name: Name of the model
        model_dir: Directory for model storage
        use_onnx: Whether to use ONNX runtime
        
    Returns:
        Loaded YOLO model
    """
    loader = ModelLoader(model_name, model_dir, use_onnx)
    return loader.load()
