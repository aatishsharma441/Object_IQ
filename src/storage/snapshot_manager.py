"""
Snapshot Manager Module
Handles saving and managing detection snapshot images.
"""

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


class SnapshotManager:
    """
    Manages detection snapshot images.
    Handles saving, organizing, and cleaning up snapshot files.
    """
    
    def __init__(
        self,
        snapshot_dir: str = "data/snapshots",
        max_age_days: int = 7,
        quality: int = 85
    ):
        """
        Initialize snapshot manager.
        
        Args:
            snapshot_dir: Directory for storing snapshots
            max_age_days: Maximum age of snapshots before cleanup
            quality: JPEG quality (1-100)
        """
        self.snapshot_dir = Path(snapshot_dir)
        self.max_age_days = max_age_days
        self.quality = quality
        
        # Ensure directory exists
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(
        self,
        frame: np.ndarray,
        object_type: str,
        confidence: float,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        timestamp: Optional[datetime] = None,
        draw_bbox: bool = True
    ) -> Optional[str]:
        """
        Save a detection snapshot.
        
        Args:
            frame: Input frame (BGR format)
            object_type: Detected object type
            confidence: Detection confidence
            bbox: Bounding box coordinates (optional)
            timestamp: Timestamp for filename (defaults to now)
            draw_bbox: Whether to draw bounding box on image
            
        Returns:
            Path to saved snapshot or None if failed
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Create date-based subdirectory
            date_dir = self.snapshot_dir / timestamp.strftime('%Y-%m-%d')
            date_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            filename = f"{object_type}_{timestamp.strftime('%H%M%S')}_{confidence:.0%}.jpg"
            filepath = date_dir / filename
            
            # Draw bounding box if provided
            if draw_bbox and bbox:
                frame = self._draw_detection(frame, object_type, confidence, bbox)
            
            # Save image
            cv2.imwrite(
                str(filepath),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving snapshot: {e}")
            return None
    
    def save_annotated_snapshot(
        self,
        frame: np.ndarray,
        detections: List[dict],
        timestamp: Optional[datetime] = None
    ) -> Optional[str]:
        """
        Save a snapshot with all detections annotated.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            timestamp: Timestamp for filename
            
        Returns:
            Path to saved snapshot
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create annotated frame
        annotated = frame.copy()
        
        for det in detections:
            bbox = det.get('bbox')
            if bbox:
                annotated = self._draw_detection(
                    annotated,
                    det.get('class_name', 'object'),
                    det.get('confidence', 0),
                    bbox
                )
        
        # Generate filename
        date_dir = self.snapshot_dir / timestamp.strftime('%Y-%m-%d')
        date_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"detection_{timestamp.strftime('%H%M%S')}.jpg"
        filepath = date_dir / filename
        
        cv2.imwrite(
            str(filepath),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        )
        
        return str(filepath)
    
    def _draw_detection(
        self,
        frame: np.ndarray,
        object_type: str,
        confidence: float,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Draw detection bounding box and label on frame.
        
        Args:
            frame: Input frame
            object_type: Object class name
            confidence: Detection confidence
            bbox: Bounding box coordinates
            
        Returns:
            Annotated frame
        """
        x1, y1, x2, y2 = bbox
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label background
        label = f"{object_type} {confidence:.0%}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            (0, 255, 0),
            -1
        )
        
        # Draw label text
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )
        
        return frame
    
    def cleanup_old_snapshots(self) -> int:
        """
        Remove snapshots older than max_age_days.
        
        Returns:
            Number of deleted files
        """
        deleted = 0
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        
        try:
            for date_dir in self.snapshot_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                
                # Try to parse directory name as date
                try:
                    dir_date = datetime.strptime(date_dir.name, '%Y-%m-%d')
                    
                    if dir_date < cutoff:
                        # Delete entire directory
                        for file in date_dir.iterdir():
                            if file.is_file():
                                file.unlink()
                                deleted += 1
                        date_dir.rmdir()
                        
                except ValueError:
                    # Not a date directory, skip
                    continue
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return deleted
    
    def get_disk_usage(self) -> dict:
        """
        Get disk usage statistics.
        
        Returns:
            Dictionary with usage statistics
        """
        total_size = 0
        file_count = 0
        
        for file_path in self.snapshot_dir.rglob('*.jpg'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
                file_count += 1
        
        return {
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'file_count': file_count,
            'snapshot_dir': str(self.snapshot_dir)
        }
    
    def get_recent_snapshots(self, limit: int = 20) -> List[str]:
        """
        Get list of recent snapshot paths.
        
        Args:
            limit: Maximum number to return
            
        Returns:
            List of snapshot file paths
        """
        snapshots = []
        
        for file_path in self.snapshot_dir.rglob('*.jpg'):
            if file_path.is_file():
                snapshots.append({
                    'path': str(file_path),
                    'mtime': file_path.stat().st_mtime
                })
        
        # Sort by modification time (newest first)
        snapshots.sort(key=lambda x: x['mtime'], reverse=True)
        
        return [s['path'] for s in snapshots[:limit]]
    
    def delete_snapshot(self, path: str) -> bool:
        """
        Delete a specific snapshot file.
        
        Args:
            path: Path to snapshot file
            
        Returns:
            True if deleted successfully
        """
        try:
            file_path = Path(path)
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                return True
        except Exception:
            pass
        return False
    
    def set_quality(self, quality: int) -> None:
        """
        Set JPEG quality for future snapshots.
        
        Args:
            quality: Quality value (1-100)
        """
        self.quality = max(1, min(100, quality))
    
    def set_max_age(self, days: int) -> None:
        """
        Set maximum age for snapshots.
        
        Args:
            days: Maximum age in days
        """
        self.max_age_days = max(1, days)
