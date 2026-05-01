"""
ROI Manager Module
Manages Region of Interest zones for detection filtering.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ROIZone:
    """
    Represents a Region of Interest zone.
    
    Attributes:
        name: Zone identifier
        coordinates: Bounding box (x1, y1, x2, y2)
        active: Whether zone is active
    """
    name: str
    coordinates: Tuple[int, int, int, int]
    active: bool = True
    
    def contains_point(self, point: Tuple[int, int]) -> bool:
        """
        Check if a point is within this zone.
        
        Args:
            point: (x, y) coordinates
            
        Returns:
            True if point is inside zone
        """
        x, y = point
        x1, y1, x2, y2 = self.coordinates
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def contains_bbox_center(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Check if bounding box center is within this zone.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            True if bbox center is inside zone
        """
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return self.contains_point((cx, cy))
    
    def get_area(self) -> int:
        """Get the area of the zone in pixels."""
        x1, y1, x2, y2 = self.coordinates
        return (x2 - x1) * (y2 - y1)


class ROIManager:
    """
    Manages multiple Region of Interest zones.
    Allows filtering detections based on geographic zones in the frame.
    """
    
    def __init__(self, zones: Optional[List[Dict]] = None):
        """
        Initialize ROI manager.
        
        Args:
            zones: List of zone dictionaries with 'name' and 'coordinates' keys
        """
        self._zones: Dict[str, ROIZone] = {}
        
        if zones:
            for zone_config in zones:
                self.add_zone(
                    name=zone_config.get('name', 'unnamed'),
                    coordinates=tuple(zone_config.get('coordinates', [0, 0, 100, 100])),
                    active=zone_config.get('active', True)
                )
    
    def add_zone(
        self,
        name: str,
        coordinates: Tuple[int, int, int, int],
        active: bool = True
    ) -> None:
        """
        Add a new ROI zone.
        
        Args:
            name: Zone identifier
            coordinates: (x1, y1, x2, y2) bounding box
            active: Whether zone is active
        """
        self._zones[name] = ROIZone(
            name=name,
            coordinates=coordinates,
            active=active
        )
    
    def remove_zone(self, name: str) -> bool:
        """
        Remove an ROI zone.
        
        Args:
            name: Zone identifier
            
        Returns:
            True if zone was removed
        """
        if name in self._zones:
            del self._zones[name]
            return True
        return False
    
    def get_zone(self, name: str) -> Optional[ROIZone]:
        """
        Get a specific zone by name.
        
        Args:
            name: Zone identifier
            
        Returns:
            ROIZone or None if not found
        """
        return self._zones.get(name)
    
    def update_zone(
        self,
        name: str,
        coordinates: Optional[Tuple[int, int, int, int]] = None,
        active: Optional[bool] = None
    ) -> bool:
        """
        Update zone properties.
        
        Args:
            name: Zone identifier
            coordinates: New coordinates (optional)
            active: New active state (optional)
            
        Returns:
            True if zone was updated
        """
        if name not in self._zones:
            return False
        
        zone = self._zones[name]
        if coordinates is not None:
            zone.coordinates = coordinates
        if active is not None:
            zone.active = active
        
        return True
    
    def toggle_zone(self, name: str) -> bool:
        """
        Toggle zone active state.
        
        Args:
            name: Zone identifier
            
        Returns:
            New active state
        """
        if name not in self._zones:
            return False
        
        self._zones[name].active = not self._zones[name].active
        return self._zones[name].active
    
    def is_in_any_zone(
        self,
        point: Tuple[int, int],
        active_only: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a point is within any ROI zone.
        
        Args:
            point: (x, y) coordinates
            active_only: Only check active zones
            
        Returns:
            Tuple of (is_in_zone, zone_name)
        """
        for name, zone in self._zones.items():
            if active_only and not zone.active:
                continue
            
            if zone.contains_point(point):
                return True, name
        
        return False, None
    
    def is_bbox_in_any_zone(
        self,
        bbox: Tuple[int, int, int, int],
        active_only: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a bounding box center is within any ROI zone.
        
        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            active_only: Only check active zones
            
        Returns:
            Tuple of (is_in_zone, zone_name)
        """
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        return self.is_in_any_zone((cx, cy), active_only)
    
    def filter_detections(
        self,
        detections: List[dict],
        active_only: bool = True
    ) -> List[dict]:
        """
        Filter detections to only include those within ROI zones.
        
        Args:
            detections: List of detection dictionaries
            active_only: Only check active zones
            
        Returns:
            Filtered list of detections
        """
        filtered = []
        
        for det in detections:
            bbox = det.get('bbox')
            if bbox is None:
                continue
            
            in_zone, zone_name = self.is_bbox_in_any_zone(bbox, active_only)
            
            if in_zone:
                det_copy = det.copy()
                det_copy['roi_zone'] = zone_name
                filtered.append(det_copy)
        
        return filtered
    
    def get_all_zones(self) -> List[ROIZone]:
        """
        Get all zones.
        
        Returns:
            List of ROIZone objects
        """
        return list(self._zones.values())
    
    def get_active_zones(self) -> List[ROIZone]:
        """
        Get active zones.
        
        Returns:
            List of active ROIZone objects
        """
        return [zone for zone in self._zones.values() if zone.active]
    
    def get_zone_count(self) -> int:
        """Get total number of zones."""
        return len(self._zones)
    
    def clear_zones(self) -> None:
        """Remove all zones."""
        self._zones.clear()
    
    def validate_coordinates(
        self,
        coordinates: Tuple[int, int, int, int],
        frame_size: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Validate and clip coordinates to frame bounds.
        
        Args:
            coordinates: (x1, y1, x2, y2)
            frame_size: (width, height)
            
        Returns:
            Clipped coordinates
        """
        x1, y1, x2, y2 = coordinates
        width, height = frame_size
        
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        # Ensure proper ordering
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        return (x1, y1, x2, y2)
    
    def to_dict(self) -> List[dict]:
        """
        Export zones to dictionary format.
        
        Returns:
            List of zone dictionaries
        """
        return [
            {
                'name': zone.name,
                'coordinates': zone.coordinates,
                'active': zone.active
            }
            for zone in self._zones.values()
        ]
