"""
Crowd Heatmap Module
Generates visual heatmaps to analyze crowd density and movement patterns.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class HeatmapConfig:
    """Configuration for heatmap generation."""
    grid_size: int = 20  # Grid cell size in pixels
    sigma: float = 15.0  # Gaussian blur sigma
    history_length: int = 100  # Number of frames to consider
    decay_rate: float = 0.95  # Decay rate for old positions
    min_intensity: float = 0.1
    max_intensity: float = 1.0
    colormap: int = cv2.COLORMAP_JET  # OpenCV colormap


class CrowdHeatmap:
    """
    Generates and manages crowd density heatmaps.
    Tracks movement patterns and visualizes crowd concentration areas.
    """
    
    # Crowd density thresholds
    DENSITY_LOW = 'low'       # < 3 people per zone
    DENSITY_MEDIUM = 'medium' # 3-7 people per zone
    DENSITY_HIGH = 'high'     # > 7 people per zone
    
    def __init__(self, config: Optional[HeatmapConfig] = None):
        """
        Initialize crowd heatmap.
        
        Args:
            config: Heatmap configuration
        """
        self.config = config or HeatmapConfig()
        
        # Accumulated heatmap
        self._heatmap: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
        
        # Position history for trajectory analysis
        self._position_history: deque = deque(maxlen=self.config.history_length)
        
        # Zone statistics
        self._zone_stats: Dict[str, Dict] = {}
        
        # Time-based analysis
        self._start_time = time.time()
        self._frame_count = 0
        
    def _init_heatmap(self, frame_shape: Tuple[int, int]):
        """Initialize heatmap array based on frame size."""
        height, width = frame_shape[:2]
        self._frame_shape = (height, width)
        self._heatmap = np.zeros((height, width), dtype=np.float32)
        
    def update(self, positions: List[Tuple[int, int]], frame: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update heatmap with new positions.
        
        Args:
            positions: List of (x, y) center positions
            frame: Optional frame to get shape from
            
        Returns:
            Current heatmap
        """
        if frame is not None and self._heatmap is None:
            self._init_heatmap(frame.shape)
            
        if self._heatmap is None:
            return np.zeros((480, 640), dtype=np.float32)
        
        # Decay existing heatmap
        self._heatmap *= self.config.decay_rate
        
        # Add new positions
        for x, y in positions:
            if 0 <= x < self._heatmap.shape[1] and 0 <= y < self._heatmap.shape[0]:
                # Add Gaussian blob at position
                self._add_gaussian_blob(x, y)
                
        # Store positions for history
        self._position_history.append({
            'positions': positions,
            'timestamp': time.time()
        })
        
        self._frame_count += 1
        
        # Update zone statistics
        self._update_zone_stats(positions)
        
        return self._heatmap.copy()
    
    def _add_gaussian_blob(self, x: int, y: int):
        """Add a Gaussian blob at the specified position."""
        height, width = self._heatmap.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.ogrid[:height, :width]
        
        # Calculate Gaussian
        sigma = self.config.sigma
        gaussian = np.exp(-((x_coords - x) ** 2 + (y_coords - y) ** 2) / (2 * sigma ** 2))
        
        # Add to heatmap
        self._heatmap += gaussian
        
        # Clip to max intensity
        self._heatmap = np.clip(self._heatmap, 0, self.config.max_intensity)
    
    def _update_zone_stats(self, positions: List[Tuple[int, int]]):
        """Update zone-based statistics."""
        if self._frame_shape is None:
            return
            
        height, width = self._frame_shape
        grid_size = self.config.grid_size
        
        # Calculate grid dimensions
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        # Reset zone counts
        zone_counts = {}
        
        for x, y in positions:
            # Determine zone
            zone_x = min(x // grid_size, grid_w - 1)
            zone_y = min(y // grid_size, grid_h - 1)
            zone_id = f"{zone_x}_{zone_y}"
            
            zone_counts[zone_id] = zone_counts.get(zone_id, 0) + 1
        
        # Update statistics
        self._zone_stats = zone_counts
    
    def get_density_level(self, count: int) -> str:
        """Get density level based on count."""
        if count < 3:
            return self.DENSITY_LOW
        elif count < 7:
            return self.DENSITY_MEDIUM
        else:
            return self.DENSITY_HIGH
    
    def get_heatmap_image(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Get heatmap overlay on frame.
        
        Args:
            frame: Input frame to overlay on
            alpha: Overlay transparency
            
        Returns:
            Frame with heatmap overlay
        """
        if self._heatmap is None:
            return frame.copy()
        
        # Normalize heatmap
        heatmap_normalized = self._heatmap / max(self._heatmap.max(), 0.001)
        heatmap_normalized = (heatmap_normalized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_normalized, self.config.colormap)
        
        # Resize heatmap to match frame size
        frame_h, frame_w = frame.shape[:2]
        heatmap_colored = cv2.resize(heatmap_colored, (frame_w, frame_h))
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def get_density_grid(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Get density grid visualization.
        
        Args:
            frame_shape: Frame dimensions
            
        Returns:
            Density grid as numpy array
        """
        height, width = frame_shape[:2]
        grid_size = self.config.grid_size
        
        grid_h = height // grid_size
        grid_w = width // grid_size
        
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for zone_id, count in self._zone_stats.items():
            try:
                x, y = map(int, zone_id.split('_'))
                density = self.get_density_level(count)
                
                # Color based on density
                if density == self.DENSITY_LOW:
                    color = (0, 255, 0)  # Green
                elif density == self.DENSITY_MEDIUM:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 0, 255)  # Red
                
                grid[y, x] = color
            except:
                pass
        
        # Scale up grid to frame size
        grid_scaled = cv2.resize(grid, (width, height), interpolation=cv2.INTER_NEAREST)
        
        return grid_scaled
    
    def get_analysis(self) -> Dict[str, Any]:
        """
        Get crowd analysis data.
        
        Returns:
            Dictionary with analysis data
        """
        # Calculate average density
        total_zones = len(self._zone_stats)
        total_people = sum(self._zone_stats.values()) if self._zone_stats else 0
        avg_density = total_people / max(total_zones, 1)
        
        # Find hotspots (zones with high density)
        hotspots = [
            {'zone': zone, 'count': count}
            for zone, count in self._zone_stats.items()
            if count >= 5
        ]
        hotspots.sort(key=lambda x: x['count'], reverse=True)
        
        # Calculate coverage (percentage of frame with people)
        coverage = 0
        if self._heatmap is not None:
            coverage = np.mean(self._heatmap > self.config.min_intensity) * 100
        
        # Time-based stats
        elapsed = time.time() - self._start_time
        fps = self._frame_count / max(elapsed, 1)
        
        return {
            'total_people': total_people,
            'total_zones': total_zones,
            'average_density': avg_density,
            'density_level': self.get_density_level(int(avg_density)),
            'hotspots': hotspots[:5],  # Top 5 hotspots
            'coverage_percent': coverage,
            'frame_count': self._frame_count,
            'elapsed_time': elapsed,
            'fps': fps,
            'zone_stats': self._zone_stats
        }
    
    def get_movement_vectors(self) -> List[Dict[str, Any]]:
        """
        Calculate movement vectors from position history.
        
        Returns:
            List of movement vectors
        """
        if len(self._position_history) < 2:
            return []
        
        vectors = []
        history = list(self._position_history)
        
        for i in range(1, len(history)):
            prev = history[i - 1]
            curr = history[i]
            
            dt = curr['timestamp'] - prev['timestamp']
            if dt == 0:
                continue
            
            for px, py in prev['positions']:
                for cx, cy in curr['positions']:
                    # Simple distance-based matching
                    dist = np.sqrt((cx - px) ** 2 + (cy - py) ** 2)
                    if dist < 50:  # Max movement per frame
                        vx = (cx - px) / dt
                        vy = (cy - py) / dt
                        vectors.append({
                            'start': (px, py),
                            'end': (cx, cy),
                            'velocity': (vx, vy),
                            'speed': np.sqrt(vx ** 2 + vy ** 2)
                        })
                        break
        
        return vectors
    
    def draw_movement_arrows(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw movement arrows on frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with movement arrows
        """
        output = frame.copy()
        vectors = self.get_movement_vectors()
        
        for vec in vectors[:20]:  # Limit arrows for clarity
            start = vec['start']
            end = vec['end']
            speed = vec['speed']
            
            # Color based on speed
            if speed < 10:
                color = (0, 255, 0)  # Slow - Green
            elif speed < 30:
                color = (0, 255, 255)  # Medium - Yellow
            else:
                color = (0, 0, 255)  # Fast - Red
            
            # Draw arrow
            cv2.arrowedLine(
                output,
                start,
                end,
                color,
                2,
                tipLength=0.3
            )
        
        return output
    
    def reset(self):
        """Reset heatmap state."""
        if self._heatmap is not None:
            self._heatmap.fill(0)
        self._position_history.clear()
        self._zone_stats = {}
        self._frame_count = 0
        self._start_time = time.time()


class CrowdAnalytics:
    """
    Comprehensive crowd analytics combining heatmap with other metrics.
    """
    
    def __init__(self, history_length: int = 100):
        """
        Initialize crowd analytics.
        
        Args:
            history_length: Number of frames to keep in history
        """
        self.heatmap = CrowdHeatmap()
        self._count_history: deque = deque(maxlen=history_length)
        self._peak_count = 0
        self._total_entries = 0
        self._total_exits = 0
        self._prev_count = 0
        
    def update(self, detections: List[Any], frame: np.ndarray) -> Dict[str, Any]:
        """
        Update analytics with new detections.
        
        Args:
            detections: List of Detection objects
            frame: Input frame
            
        Returns:
            Analytics data dictionary
        """
        # Filter people
        people = [d for d in detections if d.class_name.lower() == 'person']
        count = len(people)
        
        # Get positions
        positions = [d.center for d in people if d.center]
        
        # Update heatmap
        self.heatmap.update(positions, frame)
        
        # Track entries/exits
        if count > self._prev_count:
            self._total_entries += (count - self._prev_count)
        elif count < self._prev_count:
            self._total_exits += (self._prev_count - count)
        
        self._prev_count = count
        
        # Update peak
        self._peak_count = max(self._peak_count, count)
        
        # Store count history
        self._count_history.append({
            'count': count,
            'timestamp': time.time()
        })
        
        # Get heatmap analysis
        heatmap_analysis = self.heatmap.get_analysis()
        
        return {
            'current_count': count,
            'peak_count': self._peak_count,
            'total_entries': self._total_entries,
            'total_exits': self._total_exits,
            'average_count': np.mean([h['count'] for h in self._count_history]) if self._count_history else 0,
            'density_level': heatmap_analysis['density_level'],
            'hotspots': heatmap_analysis['hotspots'],
            'coverage': heatmap_analysis['coverage_percent'],
            'zone_stats': heatmap_analysis['zone_stats']
        }
    
    def get_visualization(self, frame: np.ndarray, show_heatmap: bool = True, show_arrows: bool = False) -> np.ndarray:
        """
        Get visualization frame.
        
        Args:
            frame: Input frame
            show_heatmap: Whether to show heatmap overlay
            show_arrows: Whether to show movement arrows
            
        Returns:
            Visualization frame
        """
        output = frame.copy()
        
        if show_heatmap:
            output = self.heatmap.get_heatmap_image(output, alpha=0.4)
        
        if show_arrows:
            output = self.heatmap.draw_movement_arrows(output)
        
        return output
    
    def get_count_history(self) -> List[Dict]:
        """Get count history for charting."""
        return list(self._count_history)
    
    def reset(self):
        """Reset all analytics."""
        self.heatmap.reset()
        self._count_history.clear()
        self._peak_count = 0
        self._total_entries = 0
        self._total_exits = 0
        self._prev_count = 0


def create_crowd_heatmap(config: Optional[HeatmapConfig] = None) -> CrowdHeatmap:
    """Factory function to create crowd heatmap."""
    return CrowdHeatmap(config=config)
