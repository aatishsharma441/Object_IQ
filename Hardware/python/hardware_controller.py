"""
Hardware Communication Module
Handles serial communication between Python object detection system and Arduino.
"""

import serial
import serial.tools.list_ports
import time
import threading
from typing import Optional


class ArduinoController:
    """
    Manages serial communication with Arduino hardware controller.
    Sends detection data and receives hardware status.
    """
    
    def __init__(self, port: Optional[str] = None, baud_rate: int = 9600):
        """
        Initialize Arduino controller.
        
        Args:
            port: Serial port (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux)
                 If None, will auto-detect
            baud_rate: Serial baud rate (default: 9600)
        """
        self.port = port
        self.baud_rate = baud_rate
        self.serial_conn: Optional[serial.Serial] = None
        self.connected = False
        self._read_thread: Optional[threading.Thread] = None
        self._running = False
        
        # State tracking
        self.last_people_count = 0
        self.last_weapon_count = 0
        self.last_alert_type = ""
        
    def auto_detect_port(self) -> Optional[str]:
        """
        Auto-detect Arduino serial port.
        
        Returns:
            Port name or None if not found
        """
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            # Look for Arduino-specific identifiers
            if 'Arduino' in port.description or 'CH340' in port.description:
                print(f"Auto-detected Arduino on {port.device}")
                return port.device
        
        # If no Arduino found, return first available port
        if ports:
            print(f"Using first available port: {ports[0].device}")
            return ports[0].device
        
        return None
    
    def connect(self, port: Optional[str] = None) -> bool:
        """
        Connect to Arduino via serial.
        
        Args:
            port: Serial port (optional, uses self.port if not provided)
            
        Returns:
            True if connected successfully
        """
        try:
            target_port = port or self.port
            
            if not target_port:
                print("Auto-detecting Arduino port...")
                target_port = self.auto_detect_port()
            
            if not target_port:
                print("ERROR: No serial port found")
                return False
            
            print(f"Connecting to Arduino on {target_port}...")
            
            self.serial_conn = serial.Serial(
                port=target_port,
                baudrate=self.baud_rate,
                timeout=1,
                write_timeout=1
            )
            
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Clear any initial garbage data
            self.serial_conn.reset_input_buffer()
            
            self.connected = True
            self.port = target_port
            
            # Start reading thread
            self._running = True
            self._read_thread = threading.Thread(
                target=self._read_loop,
                daemon=True
            )
            self._read_thread.start()
            
            print("✓ Arduino connected successfully")
            return True
            
        except serial.SerialException as e:
            print(f"ERROR: Failed to connect to Arduino: {e}")
            self.connected = False
            return False
        except Exception as e:
            print(f"ERROR: Unexpected error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino."""
        self._running = False
        
        if self.serial_conn and self.serial_conn.is_open:
            try:
                self.serial_conn.close()
                print("Arduino disconnected")
            except:
                pass
        
        self.connected = False
    
    def send_command(self, command: str):
        """
        Send command to Arduino.
        
        Args:
            command: Command string (e.g., 'PEOPLE:5', 'WEAPON:1')
        """
        if not self.connected or not self.serial_conn:
            return
        
        try:
            # Add newline terminator
            full_command = f"{command}\n"
            self.serial_conn.write(full_command.encode('utf-8'))
        except serial.SerialException as e:
            print(f"ERROR: Failed to send command: {e}")
            self.connected = False
        except Exception as e:
            print(f"ERROR: Unexpected error sending command: {e}")
    
    def update_people_count(self, count: int):
        """
        Update people count on hardware display.
        
        Args:
            count: Current people count
        """
        if count != self.last_people_count:
            self.send_command(f"PEOPLE:{count}")
            self.last_people_count = count
    
    def send_weapon_alert(self, weapon_count: int = 1, weapon_type: str = "unknown"):
        """
        Send weapon detection alert.
        
        Args:
            weapon_count: Number of weapons detected
            weapon_type: Type of weapon (gun, knife, etc.)
        """
        print(f"[HARDWARE] Sending WEAPON alert: count={weapon_count}, type={weapon_type}")
        self.send_command(f"WEAPON:{weapon_count}")
        self.last_weapon_count = weapon_count
        self.last_alert_type = f"WEAPON:{weapon_type}"
    
    def send_crowd_alert(self, crowd_level: int):
        """
        Send crowd threshold alert.
        
        Args:
            crowd_level: Alert level (1 = warning, 2 = high alert)
        """
        self.send_command(f"CROWD_ALERT:{crowd_level}")
        self.last_alert_type = f"CROWD_LEVEL:{crowd_level}"
    
    def update_status(self, status: str):
        """
        Update system status on hardware.
        
        Args:
            status: Status string (NORMAL, DETECTING, ALERT, etc.)
        """
        self.send_command(f"STATUS:{status}")
    
    def clear_alert(self):
        """Clear active alert on hardware."""
        self.send_command("CLEAR_ALERT")
        self.last_alert_type = ""
    
    def reset_hardware(self):
        """Reset hardware controller."""
        self.send_command("RESET")
        self.last_people_count = 0
        self.last_weapon_count = 0
        self.last_alert_type = ""
    
    def _read_loop(self):
        """Background thread to read responses from Arduino."""
        while self._running and self.connected:
            try:
                if self.serial_conn and self.serial_conn.in_waiting > 0:
                    response = self.serial_conn.readline().decode('utf-8').strip()
                    self._handle_response(response)
            except:
                pass
            
            time.sleep(0.1)
    
    def _handle_response(self, response: str):
        """
        Handle response from Arduino.
        
        Args:
            response: Response string from Arduino
        """
        if response:
            print(f"[Arduino] {response}")
            
            if response == "ARDUINO_READY":
                print("✓ Arduino is ready and waiting for commands")
    
    def is_connected(self) -> bool:
        """Check if Arduino is connected."""
        return self.connected and self.serial_conn and self.serial_conn.is_open
    
    def test_connection(self) -> bool:
        """
        Test connection by sending a RESET command.
        
        Returns:
            True if Arduino responds
        """
        if not self.connected:
            return False
        
        self.reset_hardware()
        time.sleep(1)
        
        return self.is_connected()
    
    def __enter__(self):
        """Context manager entry."""
        if not self.connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def __del__(self):
        """Destructor."""
        self.disconnect()


class HardwareIntegration:
    """
    High-level integration class that connects detection system to hardware.
    Use this class to integrate with the main detection system.
    """
    
    def __init__(self, arduino_port: Optional[str] = None, 
                 crowd_threshold: int = 5,
                 high_crowd_threshold: int = 10):
        """
        Initialize hardware integration.
        
        Args:
            arduino_port: Serial port for Arduino
            crowd_threshold: People count threshold for warning
            high_crowd_threshold: People count threshold for high alert
        """
        self.arduino = ArduinoController(port=arduino_port)
        self.crowd_threshold = crowd_threshold
        self.high_crowd_threshold = high_crowd_threshold
        
        # State
        self.hardware_enabled = False
        self.current_people = 0
        self.current_weapons = 0
        self.alert_active = False
    
    def enable(self) -> bool:
        """
        Enable hardware integration.
        
        Returns:
            True if successfully connected to Arduino
        """
        print("\n=== Enabling Hardware Integration ===")
        
        if self.arduino.connect():
            self.hardware_enabled = True
            self.arduino.update_status("INITIALIZED")
            print("✓ Hardware integration enabled")
            return True
        else:
            print("✗ Failed to enable hardware integration")
            return False
    
    def disable(self):
        """Disable hardware integration."""
        self.hardware_enabled = False
        self.arduino.disconnect()
        print("Hardware integration disabled")
    
    def process_detection(self, people_count: int, weapon_count: int = 0,
                         suspicious_objects: list = None):
        """
        Process detection results and update hardware.
        Call this from your main detection loop.
        
        Args:
            people_count: Number of people detected
            weapon_count: Number of weapons detected
            suspicious_objects: List of suspicious object types
        """
        if not self.hardware_enabled:
            return
        
        self.current_people = people_count
        self.current_weapons = weapon_count
        
        print(f"[HARDWARE] process_detection: people={people_count}, weapons={weapon_count}, alert_active={self.alert_active}")
        
        # Update people count
        self.arduino.update_people_count(people_count)
        
        # Check for weapon alerts (highest priority)
        if weapon_count > 0:
            print(f"[HARDWARE] Weapon count > 0, sending weapon alert")
            # Always send weapon alert when weapon detected (even if alert already active)
            self.arduino.send_weapon_alert(weapon_count)
            self.alert_active = True
        # Check for crowd alerts
        elif people_count >= self.high_crowd_threshold:
            if not self.alert_active or self.last_alert_type != "CROWD_HIGH":
                self.arduino.send_crowd_alert(2)  # High alert
                self.alert_active = True
                self.last_alert_type = "CROWD_HIGH"
        elif people_count >= self.crowd_threshold:
            if not self.alert_active or self.last_alert_type != "CROWD_WARNING":
                self.arduino.send_crowd_alert(1)  # Warning
                self.alert_active = True
                self.last_alert_type = "CROWD_WARNING"
        else:
            # No alerts - clear if was active
            if self.alert_active:
                print(f"[HARDWARE] Clearing alert - no threats detected")
                self.arduino.clear_alert()
                self.alert_active = False
                self.last_alert_type = ""
    
    def update_status(self, status: str):
        """
        Update system status on hardware.
        
        Args:
            status: Status string
        """
        if self.hardware_enabled:
            self.arduino.update_status(status)
    
    def shutdown(self):
        """Shutdown hardware integration."""
        if self.hardware_enabled:
            self.arduino.update_status("SHUTDOWN")
            time.sleep(1)
            self.disable()


# Example usage
if __name__ == "__main__":
    print("=== Hardware Integration Test ===\n")
    
    # Create controller
    controller = ArduinoController()
    
    # Try to connect
    if controller.connect():
        print("\nRunning test sequence...")
        
        # Test sequence
        controller.update_status("TESTING")
        time.sleep(1)
        
        controller.update_people_count(3)
        time.sleep(2)
        
        controller.update_people_count(7)
        controller.send_crowd_alert(1)
        time.sleep(2)
        
        controller.update_people_count(12)
        controller.send_crowd_alert(2)
        time.sleep(2)
        
        controller.send_weapon_alert(1, "gun")
        time.sleep(2)
        
        controller.clear_alert()
        controller.update_people_count(0)
        controller.update_status("NORMAL")
        
        print("\nTest complete!")
        controller.disconnect()
    else:
        print("Failed to connect to Arduino")
