"""
Configuration Loader Module
Handles loading and validation of YAML configuration files.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigLoader:
    """Loads and manages configuration from YAML files."""
    
    def __init__(self, config_path: Optional[str] = None, objects_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to main config.yaml
            objects_path: Path to objects.yaml
        """
        self.base_dir = Path(__file__).parent.parent  # Object_Detact directory
        
        # Default paths
        self.config_path = Path(config_path) if config_path else self.base_dir / "config" / "config.yaml"
        self.objects_path = Path(objects_path) if objects_path else self.base_dir / "config" / "objects.yaml"
        
        # Configuration data
        self._config: Dict[str, Any] = {}
        self._objects_config: Dict[str, Any] = {}
        
        # Load configurations
        self._load_config()
        self._load_objects_config()
    
    def _load_config(self) -> None:
        """Load main configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f) or {}
    
    def _load_objects_config(self) -> None:
        """Load objects configuration from YAML file."""
        if not self.objects_path.exists():
            raise FileNotFoundError(f"Objects configuration file not found: {self.objects_path}")
        
        with open(self.objects_path, 'r', encoding='utf-8') as f:
            self._objects_config = yaml.safe_load(f) or {}
    
    def reload(self) -> None:
        """Reload configuration from files."""
        self._load_config()
        self._load_objects_config()
    
    # Detection Settings
    @property
    def detection(self) -> Dict[str, Any]:
        """Get detection configuration."""
        return self._config.get('detection', {})
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.detection.get('model', 'yolov8n.pt')
    
    @property
    def confidence_threshold(self) -> float:
        """Get default confidence threshold."""
        return self.detection.get('confidence_threshold', 0.70)
    
    @property
    def frame_rate(self) -> int:
        """Get target frame rate."""
        return self.detection.get('frame_rate', 15)
    
    @property
    def resolution(self) -> tuple:
        """Get target resolution as (width, height)."""
        res = self.detection.get('resolution', [640, 480])
        return tuple(res)
    
    @property
    def use_onnx(self) -> bool:
        """Check if ONNX runtime should be used."""
        return self.detection.get('use_onnx', False)
    
    # Temporal Filtering
    @property
    def temporal(self) -> Dict[str, Any]:
        """Get temporal filtering configuration."""
        return self._config.get('temporal', {})
    
    @property
    def consecutive_frames(self) -> int:
        """Get number of consecutive frames required."""
        return self.temporal.get('consecutive_frames', 3)
    
    # Cooldown Settings
    @property
    def cooldown(self) -> Dict[str, Any]:
        """Get cooldown configuration."""
        return self._config.get('cooldown', {})
    
    @property
    def cooldown_seconds(self) -> int:
        """Get default cooldown duration in seconds."""
        return self.cooldown.get('seconds', 60)
    
    # ROI Settings
    @property
    def roi(self) -> Dict[str, Any]:
        """Get ROI configuration."""
        return self._config.get('roi', {})
    
    @property
    def roi_enabled(self) -> bool:
        """Check if ROI is enabled."""
        return self.roi.get('enabled', False)
    
    @property
    def roi_zones(self) -> List[Dict[str, Any]]:
        """Get ROI zones."""
        return self.roi.get('zones', [])
    
    # Object Thresholds
    @property
    def object_thresholds(self) -> Dict[str, Any]:
        """Get object threshold configuration."""
        return self._config.get('object_thresholds', {})
    
    @property
    def object_thresholds_enabled(self) -> bool:
        """Check if object thresholds are enabled."""
        return self.object_thresholds.get('enabled', False)
    
    # Notification Settings
    @property
    def notifications(self) -> Dict[str, Any]:
        """Get notification configuration."""
        return self._config.get('notifications', {})
    
    @property
    def email_config(self) -> Dict[str, Any]:
        """Get email notification configuration."""
        return self.notifications.get('email', {})
    
    @property
    def email_enabled(self) -> bool:
        """Check if email notifications are enabled."""
        return self.email_config.get('enabled', False)
    
    # Camera Settings
    @property
    def camera(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self._config.get('camera', {})
    
    @property
    def camera_source(self) -> Any:
        """Get camera source (index, URL, or path)."""
        source = self.camera.get('source', 0)
        # Try to convert to int if it's a numeric string
        if isinstance(source, str) and source.isdigit():
            return int(source)
        return source
    
    @property
    def reconnect_delay(self) -> int:
        """Get reconnection delay in seconds."""
        return self.camera.get('reconnect_delay', 5)
    
    # Logging Settings
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    @property
    def log_level(self) -> str:
        """Get logging level."""
        return self.logging_config.get('level', 'INFO')
    
    @property
    def save_snapshots(self) -> bool:
        """Check if snapshots should be saved."""
        return self.logging_config.get('save_snapshots', True)
    
    @property
    def snapshot_dir(self) -> Path:
        """Get snapshot directory path."""
        dir_path = self.logging_config.get('snapshot_dir', 'data/snapshots')
        return self.base_dir / dir_path
    
    @property
    def log_dir(self) -> Path:
        """Get log directory path."""
        dir_path = self.logging_config.get('log_dir', 'data/logs')
        return self.base_dir / dir_path
    
    # Database Settings (MongoDB Atlas)
    @property
    def database(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self._config.get('database', {})
        
    @property
    def mongodb_connection_string(self) -> str:
        """Get MongoDB Atlas connection string."""
        return self.database.get('connection_string', '')
        
    @property
    def mongodb_database_name(self) -> str:
        """Get MongoDB database name."""
        return self.database.get('database_name', 'object_detection')
        
    @property
    def mongodb_collections(self) -> Dict[str, str]:
        """Get MongoDB collection names."""
        return self.database.get('collections', {
            'detections': 'detections',
            'alerts': 'alerts'
        })
        
    @property
    def mongodb_options(self) -> Dict[str, Any]:
        """Get MongoDB connection options."""
        return self.database.get('options', {})
        
    @property
    def cache_enabled(self) -> bool:
        """Check if local caching is enabled."""
        return self.database.get('cache_enabled', True)
        
    @property
    def cache_max_size_mb(self) -> int:
        """Get maximum cache size in MB."""
        return self.database.get('cache_max_size_mb', 100)
    
    # Objects Configuration
    @property
    def target_objects(self) -> List[str]:
        """Get list of target objects to detect."""
        return self._objects_config.get('target_objects', [])
    
    @property
    def object_config(self) -> Dict[str, Dict[str, Any]]:
        """Get object-specific configuration."""
        return self._objects_config.get('object_config', {})
    
    @property
    def priority_levels(self) -> Dict[str, Dict[str, Any]]:
        """Get priority level definitions."""
        return self._objects_config.get('priority_levels', {})
    
    def get_object_confidence(self, object_name: str) -> float:
        """
        Get confidence threshold for specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Confidence threshold (object-specific or global default)
        """
        obj_config = self.object_config.get(object_name, {})
        return obj_config.get('confidence_override', self.confidence_threshold)
    
    def get_object_cooldown(self, object_name: str) -> int:
        """
        Get cooldown duration for specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Cooldown duration in seconds (object-specific or global default)
        """
        obj_config = self.object_config.get(object_name, {})
        return obj_config.get('cooldown_override', self.cooldown_seconds)
    
    def get_object_priority(self, object_name: str) -> str:
        """
        Get priority level for specific object.
        
        Args:
            object_name: Name of the object
            
        Returns:
            Priority level string
        """
        obj_config = self.object_config.get(object_name, {})
        return obj_config.get('priority', 'medium')
    
    def get_priority_config(self, priority: str) -> Dict[str, Any]:
        """
        Get configuration for a priority level.
        
        Args:
            priority: Priority level name
            
        Returns:
            Priority configuration dictionary
        """
        return self.priority_levels.get(priority, {})
    
    def is_target_object(self, object_name: str) -> bool:
        """
        Check if object is in the target list.
        
        Args:
            object_name: Name of the object
            
        Returns:
            True if object is a target
        """
        return object_name.lower() in [obj.lower() for obj in self.target_objects]
    
    def get_model_path(self) -> Path:
        """
        Get the full path to the model file.
        
        Returns:
            Path to model file
        """
        model_name = self.model_name
        if os.path.isabs(model_name):
            return Path(model_name)
        return self.base_dir / "models" / model_name


# Global configuration instance
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: Optional[str] = None, objects_path: Optional[str] = None) -> ConfigLoader:
    """
    Get or create the global configuration instance.
    
    Args:
        config_path: Optional path to config.yaml
        objects_path: Optional path to objects.yaml
        
    Returns:
        ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path, objects_path)
    return _config_instance


def reload_config() -> None:
    """Reload the global configuration."""
    global _config_instance
    if _config_instance:
        _config_instance.reload()
