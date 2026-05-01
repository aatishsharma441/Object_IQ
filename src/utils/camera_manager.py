"""
Camera Manager Module
Handles camera/video input source management.
"""

import threading
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np


class CameraState(Enum):
    """Camera state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class CameraManager:
    """
    Manages camera/video input sources.
    Supports USB webcams, RTSP streams, and video files.
    Provides automatic reconnection and thread-safe frame access.
    """
    
    def __init__(
        self,
        source: Union[int, str] = 0,
        reconnect_delay: int = 5,
        frame_rate: Optional[int] = None,
        resolution: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize camera manager.
        
        Args:
            source: Camera index, RTSP URL, or video file path
            reconnect_delay: Seconds to wait before reconnecting
            frame_rate: Target frame rate (None = camera default)
            resolution: Target resolution (width, height)
        """
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.target_frame_rate = frame_rate
        self.target_resolution = resolution
        
        # Camera state
        self._capture: Optional[cv2.VideoCapture] = None
        self._state = CameraState.DISCONNECTED
        self._last_error: Optional[str] = None
        
        # Threading
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._frame_lock = threading.Lock()
        self._current_frame: Optional[np.ndarray] = None
        self._frame_number = 0
        
        # Callbacks
        self._on_frame_callback: Optional[Callable] = None
        self._on_error_callback: Optional[Callable] = None
    
    def connect(self) -> bool:
        """
        Connect to the camera source.
        
        Returns:
            True if connected successfully
        """
        self._state = CameraState.CONNECTING
        
        try:
            # Create capture object based on source type
            if isinstance(self.source, str):
                if self.source.startswith(('rtsp://', 'http://', 'https://')):
                    # Network stream
                    self._capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                elif Path(self.source).exists():
                    # Video file
                    self._capture = cv2.VideoCapture(self.source)
                else:
                    raise ValueError(f"Invalid source: {self.source}")
            else:
                # USB camera by index
                self._capture = cv2.VideoCapture(self.source)
            
            # Check if opened successfully
            if not self._capture.isOpened():
                raise RuntimeError("Failed to open camera source")
            
            # Configure camera properties
            if self.target_resolution:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_resolution[0])
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_resolution[1])
            
            if self.target_frame_rate:
                self._capture.set(cv2.CAP_PROP_FPS, self.target_frame_rate)
            
            self._state = CameraState.CONNECTED
            self._last_error = None
            return True
            
        except Exception as e:
            self._state = CameraState.ERROR
            self._last_error = str(e)
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the camera source."""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        
        if self._capture:
            self._capture.release()
            self._capture = None
        
        self._state = CameraState.DISCONNECTED
    
    def start_capture(self) -> bool:
        """
        Start continuous frame capture in a background thread.
        
        Returns:
            True if started successfully
        """
        if not self.connect():
            return False
        
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return True
    
    def stop_capture(self) -> None:
        """Stop continuous frame capture."""
        self.disconnect()
    
    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        frame_interval = 1.0 / self.target_frame_rate if self.target_frame_rate else 0
        
        while self._running:
            start_time = time.time()
            
            try:
                # Read frame
                ret, frame = self._capture.read()
                
                if not ret or frame is None:
                    # Frame read failed
                    if self._on_error_callback:
                        self._on_error_callback("Frame read failed")
                    
                    # Attempt reconnection
                    self._attempt_reconnection()
                    continue
                
                # Update current frame
                with self._frame_lock:
                    self._current_frame = frame.copy()
                    self._frame_number += 1
                
                # Call frame callback if set
                if self._on_frame_callback:
                    self._on_frame_callback(frame, self._frame_number)
                
                # Maintain frame rate
                if frame_interval > 0:
                    elapsed = time.time() - start_time
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        
            except Exception as e:
                self._last_error = str(e)
                self._state = CameraState.ERROR
                
                if self._on_error_callback:
                    self._on_error_callback(str(e))
                
                self._attempt_reconnection()
    
    def _attempt_reconnection(self) -> None:
        """Attempt to reconnect to the camera."""
        self._state = CameraState.CONNECTING
        
        if self._capture:
            self._capture.release()
        
        time.sleep(self.reconnect_delay)
        
        if self.connect():
            print("Reconnected to camera")
        else:
            print(f"Reconnection failed: {self._last_error}")
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame (synchronous).
        
        Returns:
            Tuple of (success, frame)
        """
        if self._state != CameraState.CONNECTED:
            if not self.connect():
                return False, None
        
        if self._capture is None:
            return False, None
        
        ret, frame = self._capture.read()
        
        if ret and frame is not None:
            self._current_frame = frame.copy()
            return True, frame
        
        return False, None
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame from the buffer.
        
        Returns:
            Current frame or None
        """
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None
    
    def get_frame_number(self) -> int:
        """Get current frame number."""
        return self._frame_number
    
    @property
    def state(self) -> CameraState:
        """Get camera state."""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if camera is connected."""
        return self._state == CameraState.CONNECTED
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
    
    def get_camera_info(self) -> dict:
        """
        Get camera information.
        
        Returns:
            Camera info dictionary
        """
        if not self._capture or not self._capture.isOpened():
            return {
                'connected': False,
                'source': str(self.source),
                'state': self._state.value
            }
        
        return {
            'connected': True,
            'source': str(self.source),
            'state': self._state.value,
            'width': int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self._capture.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            'backend': self._capture.getBackendName()
        }
    
    def set_on_frame_callback(self, callback: Callable[[np.ndarray, int], None]) -> None:
        """
        Set callback for new frames.
        
        Args:
            callback: Function(frame, frame_number)
        """
        self._on_frame_callback = callback
    
    def set_on_error_callback(self, callback: Callable[[str], None]) -> None:
        """
        Set callback for errors.
        
        Args:
            callback: Function(error_message)
        """
        self._on_error_callback = callback
    
    def set_frame_rate(self, fps: int) -> None:
        """
        Set target frame rate.
        
        Args:
            fps: Frames per second
        """
        self.target_frame_rate = fps
        
        if self._capture and self._capture.isOpened():
            self._capture.set(cv2.CAP_PROP_FPS, fps)
    
    def set_resolution(self, width: int, height: int) -> None:
        """
        Set target resolution.
        
        Args:
            width: Frame width
            height: Frame height
        """
        self.target_resolution = (width, height)
        
        if self._capture and self._capture.isOpened():
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Destructor."""
        self.disconnect()
