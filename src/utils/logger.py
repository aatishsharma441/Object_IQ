"""
Logging Utilities Module
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "object_detection",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None = no file logging)
        console_output: Whether to output to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log file with date
        log_file = log_dir / f"detection_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class DetectionLogger:
    """
    Specialized logger for detection events.
    Provides structured logging for detection-related information.
    """
    
    def __init__(self, name: str = "detection", log_dir: Optional[Path] = None, level: str = "INFO"):
        """
        Initialize detection logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.logger = setup_logger(name, level, log_dir)
    
    def log_detection(
        self,
        object_type: str,
        confidence: float,
        bbox: tuple,
        frame_number: Optional[int] = None
    ) -> None:
        """
        Log a detection event.
        
        Args:
            object_type: Type of detected object
            confidence: Detection confidence score
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            frame_number: Optional frame number
        """
        frame_info = f"Frame {frame_number} | " if frame_number is not None else ""
        self.logger.info(
            f"{frame_info}Detected '{object_type}' with confidence {confidence:.2%} "
            f"at bbox={bbox}"
        )
    
    def log_alert(
        self,
        object_type: str,
        confidence: float,
        consecutive_frames: int,
        notification_sent: bool = True
    ) -> None:
        """
        Log an alert event.
        
        Args:
            object_type: Type of detected object
            confidence: Detection confidence score
            consecutive_frames: Number of consecutive frames detected
            notification_sent: Whether notification was sent
        """
        notif_status = "sent" if notification_sent else "skipped"
        self.logger.warning(
            f"ALERT: '{object_type}' detected for {consecutive_frames} consecutive frames "
            f"(confidence: {confidence:.2%}) | Notification {notif_status}"
        )
    
    def log_cooldown(self, object_type: str, remaining_seconds: int) -> None:
        """
        Log cooldown status.
        
        Args:
            object_type: Type of object in cooldown
            remaining_seconds: Remaining cooldown time
        """
        self.logger.debug(
            f"'{object_type}' in cooldown for {remaining_seconds}s"
        )
    
    def log_temporal_progress(
        self,
        object_type: str,
        current_count: int,
        required_count: int
    ) -> None:
        """
        Log temporal filter progress.
        
        Args:
            object_type: Type of object being tracked
            current_count: Current consecutive detection count
            required_count: Required consecutive detections
        """
        self.logger.debug(
            f"'{object_type}' temporal progress: {current_count}/{required_count} frames"
        )
    
    def log_camera_event(self, event: str, details: str = "") -> None:
        """
        Log camera-related events.
        
        Args:
            event: Event type (connected, disconnected, reconnecting)
            details: Additional details
        """
        detail_str = f" | {details}" if details else ""
        self.logger.info(f"Camera {event}{detail_str}")
    
    def log_notification(
        self,
        channel: str,
        status: str,
        recipient: str,
        error: Optional[str] = None
    ) -> None:
        """
        Log notification event.
        
        Args:
            channel: Notification channel (email, sms, etc.)
            status: Status (sent, failed)
            recipient: Recipient address/number
            error: Error message if failed
        """
        if status == "sent":
            self.logger.info(f"Notification ({channel}) sent to {recipient}")
        else:
            self.logger.error(
                f"Notification ({channel}) failed to {recipient}: {error}"
            )
    
    def log_error(self, component: str, error: Exception) -> None:
        """
        Log an error event.
        
        Args:
            component: Component where error occurred
            error: Exception object
        """
        self.logger.error(f"Error in {component}: {type(error).__name__}: {error}")
    
    def log_startup(self, config_summary: dict) -> None:
        """
        Log system startup information.
        
        Args:
            config_summary: Summary of loaded configuration
        """
        self.logger.info("=" * 60)
        self.logger.info("Object Detection System Starting")
        self.logger.info("=" * 60)
        for key, value in config_summary.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_shutdown(self, reason: str = "User requested") -> None:
        """
        Log system shutdown.
        
        Args:
            reason: Reason for shutdown
        """
        self.logger.info(f"System shutting down: {reason}")
