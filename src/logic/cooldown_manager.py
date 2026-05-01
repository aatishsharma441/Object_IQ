"""
Cooldown Manager Module
Manages cooldown periods between notifications to prevent spam.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CooldownState:
    """
    Represents the cooldown state for an object type.
    
    Attributes:
        last_alert_time: Timestamp of last alert
        cooldown_seconds: Cooldown duration in seconds
    """
    last_alert_time: float
    cooldown_seconds: int


class CooldownManager:
    """
    Manages cooldown periods for alerts.
    Prevents repeated notifications for the same object type within
    a configurable time window.
    """
    
    def __init__(self, default_cooldown_seconds: int = 60):
        """
        Initialize cooldown manager.
        
        Args:
            default_cooldown_seconds: Default cooldown duration
        """
        self.default_cooldown_seconds = default_cooldown_seconds
        
        # Cooldown state per object type
        self._cooldown_states: Dict[str, CooldownState] = {}
        
        # Object-specific cooldown overrides
        self._cooldown_overrides: Dict[str, int] = {}
    
    def set_cooldown_override(
        self,
        object_type: str,
        cooldown_seconds: int
    ) -> None:
        """
        Set a custom cooldown duration for a specific object type.
        
        Args:
            object_type: Object class name
            cooldown_seconds: Custom cooldown duration
        """
        self._cooldown_overrides[object_type.lower()] = cooldown_seconds
    
    def get_cooldown_duration(self, object_type: str) -> int:
        """
        Get the cooldown duration for an object type.
        
        Args:
            object_type: Object class name
            
        Returns:
            Cooldown duration in seconds
        """
        return self._cooldown_overrides.get(
            object_type.lower(),
            self.default_cooldown_seconds
        )
    
    def is_on_cooldown(self, object_type: str) -> bool:
        """
        Check if an object type is currently on cooldown.
        
        Args:
            object_type: Object class name
            
        Returns:
            True if object is on cooldown
        """
        object_type = object_type.lower()
        
        if object_type not in self._cooldown_states:
            return False
        
        state = self._cooldown_states[object_type]
        elapsed = time.time() - state.last_alert_time
        
        return elapsed < state.cooldown_seconds
    
    def get_remaining_cooldown(self, object_type: str) -> int:
        """
        Get remaining cooldown time for an object type.
        
        Args:
            object_type: Object class name
            
        Returns:
            Remaining cooldown seconds (0 if not on cooldown)
        """
        object_type = object_type.lower()
        
        if object_type not in self._cooldown_states:
            return 0
        
        state = self._cooldown_states[object_type]
        elapsed = time.time() - state.last_alert_time
        remaining = state.cooldown_seconds - int(elapsed)
        
        return max(0, remaining)
    
    def start_cooldown(self, object_type: str) -> None:
        """
        Start cooldown for an object type after an alert.
        
        Args:
            object_type: Object class name
        """
        object_type = object_type.lower()
        cooldown_duration = self.get_cooldown_duration(object_type)
        
        self._cooldown_states[object_type] = CooldownState(
            last_alert_time=time.time(),
            cooldown_seconds=cooldown_duration
        )
    
    def reset_cooldown(self, object_type: str) -> None:
        """
        Reset cooldown for an object type.
        
        Args:
            object_type: Object class name
        """
        object_type = object_type.lower()
        
        if object_type in self._cooldown_states:
            del self._cooldown_states[object_type]
    
    def reset_all(self) -> None:
        """Reset all cooldown states."""
        self._cooldown_states.clear()
    
    def get_all_cooldowns(self) -> Dict[str, dict]:
        """
        Get cooldown status for all object types.
        
        Returns:
            Dictionary mapping object types to cooldown info
        """
        result = {}
        
        for object_type, state in self._cooldown_states.items():
            elapsed = time.time() - state.last_alert_time
            remaining = max(0, state.cooldown_seconds - int(elapsed))
            
            result[object_type] = {
                'on_cooldown': remaining > 0,
                'remaining_seconds': remaining,
                'total_seconds': state.cooldown_seconds
            }
        
        return result
    
    def cleanup_expired(self) -> List[str]:
        """
        Remove expired cooldown entries.
        
        Returns:
            List of object types that were cleaned up
        """
        expired = []
        current_time = time.time()
        
        for object_type, state in list(self._cooldown_states.items()):
            elapsed = current_time - state.last_alert_time
            if elapsed >= state.cooldown_seconds:
                expired.append(object_type)
                del self._cooldown_states[object_type]
        
        return expired
    
    def set_default_cooldown(self, seconds: int) -> None:
        """
        Set the default cooldown duration.
        
        Args:
            seconds: Default cooldown in seconds
        """
        self.default_cooldown_seconds = max(0, seconds)
    
    def get_objects_on_cooldown(self) -> List[str]:
        """
        Get list of object types currently on cooldown.
        
        Returns:
            List of object type names
        """
        return [
            obj_type for obj_type in self._cooldown_states
            if self.is_on_cooldown(obj_type)
        ]
