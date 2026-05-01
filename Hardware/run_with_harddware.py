"""
Object Detection System with Hardware Integration
Runs the main detection system and sends real-time data to Arduino hardware.

This script:
1. Loads the original detection system
2. Connects to Arduino via serial
3. Sends people count, weapon alerts, and crowd data to hardware
4. Displays everything on LCD + LEDs + Buzzer
"""

import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Add Hardware/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

# Add main project src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hardware_controller import HardwareIntegration
from src.run_dashboard import DashboardDetectionSystem
from src.web import update_frame, add_crowd_data, update_statistics
import time


class DetectionWithHardware(DashboardDetectionSystem):
    """
    Extended detection system with hardware output.
    Inherits from original DashboardDetectionSystem and adds hardware integration.
    """
    
    def __init__(self, config_path=None, dashboard_port=5000, 
                 arduino_port=None, crowd_threshold=5, high_crowd_threshold=10):
        """
        Initialize detection system with hardware.
        
        Args:
            config_path: Path to config.yaml
            dashboard_port: Web dashboard port
            arduino_port: Serial port for Arduino (None = auto-detect)
            crowd_threshold: People count for warning
            high_crowd_threshold: People count for high alert
        """
        # Initialize parent (original system)
        super().__init__(config_path=config_path, dashboard_port=dashboard_port)
        
        # Initialize hardware integration
        self.hardware = HardwareIntegration(
            arduino_port=arduino_port,
            crowd_threshold=crowd_threshold,
            high_crowd_threshold=high_crowd_threshold
        )
        
        # Tracking variables
        self.last_hardware_update = 0
        self.hardware_update_interval = 1.0  # Update hardware every 1 second
        
        # Weapon detection tracking for hardware
        self._recent_weapon_detections = []
        self._weapon_detection_cooldown = 5.0  # seconds to keep weapon alert active
        self._last_weapon_detection_time = 0
    
    def initialize(self) -> bool:
        """Initialize both detection system and hardware."""
        print("\n" + "="*60)
        print("INITIALIZING OBJECT DETECTION SYSTEM WITH HARDWARE")
        print("="*60 + "\n")
        
        # Initialize original system
        if not super().initialize():
            print("✗ Failed to initialize detection system")
            return False
        
        # Initialize hardware
        print("\n--- Hardware Initialization ---")
        if not self.hardware.enable():
            print("⚠ Hardware not available - continuing without hardware")
            print("   The software will run normally, just no LCD/LED/Buzzer output\n")
        else:
            print("✓ Hardware integration enabled\n")
        
        return True
    
    def _handle_weapon_detection(self, frame, detection, severity, person_nearby, crowd_count):
        """
        Override parent method to track weapon detections for hardware alerts.
        """
        print(f"\n[WEAPON_OVERRIDE] _handle_weapon_detection CALLED! Weapon: {detection.class_name}, Severity: {severity}\n")
        
        # Call parent method to handle email, snapshots, database, etc.
        super()._handle_weapon_detection(frame, detection, severity, person_nearby, crowd_count)
        
        # Track weapon detection for hardware
        current_time = time.time()
        self._last_weapon_detection_time = current_time
        
        # Add to recent detections
        self._recent_weapon_detections.append({
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'severity': severity,
            'timestamp': current_time
        })
        
        print(f"[WEAPON_OVERRIDE] Added to recent detections. Total: {len(self._recent_weapon_detections)}")
        
        # Keep only recent detections (within cooldown period)
        self._recent_weapon_detections = [
            d for d in self._recent_weapon_detections 
            if current_time - d['timestamp'] < self._weapon_detection_cooldown
        ]
    
    def _handle_suspicious_activity(self, frame, detection, severity='medium', person_nearby=False, crowd_count=0):
        """
        Override parent method to track suspicious activity for hardware alerts.
        """
        # Call parent method to handle email, snapshots, database, etc.
        super()._handle_suspicious_activity(frame, detection, severity, person_nearby, crowd_count)
        
        # Track suspicious detection for hardware (treat as weapon for hardware)
        current_time = time.time()
        self._last_weapon_detection_time = current_time
        
        # Add to recent detections
        self._recent_weapon_detections.append({
            'class_name': detection.class_name,
            'confidence': detection.confidence,
            'severity': severity,
            'timestamp': current_time
        })
        
        # Keep only recent detections (within cooldown period)
        self._recent_weapon_detections = [
            d for d in self._recent_weapon_detections 
            if current_time - d['timestamp'] < self._weapon_detection_cooldown
        ]
    
    def _update_hardware(self, detections):
        """
        Send detection data to hardware.
        
        Args:
            detections: List of detection objects
        """
        try:
            # Count people
            people_count = sum(1 for d in detections if d.class_name.lower() == 'person')
            
            # Count weapons (if weapon detector is enabled)
            weapon_count = 0
            if hasattr(self, 'weapon_detector'):
                weapon_count = sum(1 for d in detections 
                                 if hasattr(d, 'is_weapon') and d.is_weapon)
            
            # Check for suspicious objects
            suspicious = []
            if hasattr(self, 'suspicious_objects'):
                suspicious = [d.class_name for d in detections 
                            if d.class_name.lower() in self.suspicious_objects]
            
            # Send to hardware
            self.hardware.process_detection(
                people_count=people_count,
                weapon_count=weapon_count,
                suspicious_objects=suspicious
            )
            
            # Update status
            if self._running:
                self.hardware.update_status("DETECTING")
            
        except Exception as e:
            print(f"Hardware update error: {e}")
    
    def _update_dashboard_stats(self):
        """Update dashboard statistics."""
        if self.database:
            stats = self.database.get_detection_stats()
            update_statistics(stats)
    
    def run(self) -> None:
        """Run the detection system with hardware integration."""
        if not self.initialize():
            print("Failed to initialize system")
            return
        
        # Start dashboard
        self.start_dashboard()
        
        # Connect to camera
        if not self.camera.connect():
            self.logger.logger.error("Failed to connect to camera")
            return
        
        self._running = True
        self.logger.logger.info("Starting detection loop with hardware...")
        
        frame_interval = 1.0 / self.config.frame_rate
        frame_count = 0
        stats_update_interval = 5.0
        last_stats_update = time.time()
        
        try:
            while self._running:
                frame_start = time.time()
                
                # Check for pending source switch
                with self._source_lock:
                    if self._pending_source is not None:
                        self.logger.logger.info(f"Switching to source: {self._pending_source}")
                        self.camera.disconnect()
                        self.camera.source = self._pending_source
                        if self.camera.connect():
                            self.logger.logger.info("Source switch successful")
                        else:
                            self.logger.logger.error("Source switch failed")
                        self._pending_source = None
                
                # Read frame
                ret, frame = self.camera.read_frame()
                if not ret:
                    self.logger.logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                # Process frame (includes hardware update, notifications, snapshots)
                processed_frame, crowd_count = self._process_frame(frame)
                
                # Update dashboard frame with crowd count
                update_frame(processed_frame, crowd_count=crowd_count)
                
                # Update crowd data periodically
                if time.time() - self._last_crowd_update >= 2.0:
                    add_crowd_data(crowd_count)
                    self._last_crowd_update = time.time()
                
                # Update hardware
                if self.hardware.hardware_enabled:
                    current_time = time.time()
                    if current_time - self.last_hardware_update >= self.hardware_update_interval:
                        # Count weapons from recent detections
                        weapon_count = len(self._recent_weapon_detections)
                        
                        # Get suspicious object types from recent detections
                        suspicious_objects = list(set([
                            d['class_name'] for d in self._recent_weapon_detections
                        ]))
                        
                        # DEBUG: Print weapon detection info
                        print(f"\n[HARDWARE_UPDATE] Checking hardware update...")
                        print(f"[HARDWARE_UPDATE] Recent weapon detections: {len(self._recent_weapon_detections)}")
                        print(f"[HARDWARE_UPDATE] Weapon count: {weapon_count}")
                        print(f"[HARDWARE_UPDATE] Suspicious objects: {suspicious_objects}")
                        print(f"[HARDWARE_UPDATE] Alert active: {self.hardware.alert_active}\n")
                        
                        if weapon_count > 0:
                            print(f"\n[DEBUG] WEAPON DETECTED! Count: {weapon_count}")
                            print(f"[DEBUG] Weapons: {suspicious_objects}")
                            print(f"[DEBUG] Sending to hardware...\n")
                        
                        # Send to hardware
                        self.hardware.process_detection(
                            people_count=crowd_count,
                            weapon_count=weapon_count,
                            suspicious_objects=suspicious_objects
                        )
                        
                        # Clean up old weapon detections
                        self._recent_weapon_detections = [
                            d for d in self._recent_weapon_detections 
                            if current_time - d['timestamp'] < self._weapon_detection_cooldown
                        ]
                        
                        self.last_hardware_update = current_time
                else:
                    # Print this only once every 100 frames to avoid spam
                    if frame_count % 100 == 0:
                        print(f"[WARNING] Hardware NOT enabled!")
                
                # Update frame count
                frame_count += 1
                
                # Update statistics periodically
                if time.time() - last_stats_update >= stats_update_interval:
                    self._update_dashboard_stats()
                    last_stats_update = time.time()
                
                # Log progress
                if frame_count % 100 == 0:
                    self.logger.logger.info(f"Processed {frame_count} frames")
                
                # Maintain frame rate
                frame_time = time.time() - frame_start
                sleep_time = frame_interval - frame_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            self.logger.logger.info("Stopping detection system...")
        except Exception as e:
            self.logger.logger.error(f"Detection error: {e}")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Clean up resources."""
        self._running = False
        
        # Shutdown hardware
        if self.hardware.hardware_enabled:
            print("\nShutting down hardware...")
            self.hardware.shutdown()
        
        # Call parent cleanup
        if hasattr(self, 'camera'):
            self.camera.disconnect()
        
        if hasattr(self, 'database'):
            self.database.close()
        
        print("System shutdown complete")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Object Detection with Hardware Integration'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=5000, 
        help='Dashboard port'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=None, 
        help='Config file path'
    )
    parser.add_argument(
        '--arduino-port', 
        type=str, 
        default=None, 
        help='Arduino serial port (e.g., COM3)'
    )
    parser.add_argument(
        '--crowd-threshold', 
        type=int, 
        default=5, 
        help='People count for crowd warning'
    )
    parser.add_argument(
        '--high-crowd-threshold', 
        type=int, 
        default=10, 
        help='People count for high alert'
    )
    
    args = parser.parse_args()
    
    # Create and run system
    system = DetectionWithHardware(
        config_path=args.config,
        dashboard_port=args.port,
        arduino_port=args.arduino_port,
        crowd_threshold=args.crowd_threshold,
        high_crowd_threshold=args.high_crowd_threshold
    )
    
    system.run()


if __name__ == '__main__':
    main()
