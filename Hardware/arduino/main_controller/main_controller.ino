/*
 * Object Detection System - Hardware Controller
 * Arduino Uno R3 with 16-pin LCD, LEDs, and Buzzer
 * 
 * Features:
 * - People counting display
 * - Weapon detection alerts
 * - Crowd threshold warnings
 * - System status monitoring
 * - Serial communication with Python
 * 
 * Components:
 * - Arduino Uno R3
 * - LCD 16x2 (16-pin, no potentiometer)
 * - Green LED (Pin 8)
 * - Red LED (Pin 7)
 * - Buzzer (Pin 9)
 * 
 * Pin Configuration:
 * LCD: RS=12, E=11, D4=5, D5=4, D6=3, D7=2
 * Green LED: Pin 8
 * Red LED: Pin 7
 * Buzzer: Pin 9
 */

#include <LiquidCrystal.h>

// ==================== Pin Definitions ====================
// LCD pins
const int LCD_RS = 12;
const int LCD_E = 11;
const int LCD_D4 = 5;
const int LCD_D5 = 4;
const int LCD_D6 = 3;
const int LCD_D7 = 2;

// Output pins
const int GREEN_LED = 8;
const int RED_LED = 7;
const int BUZZER = 9;

// IR Sensor pins
const int IR_ENTRY = A0;    // Entry sensor
const int IR_EXIT = A1;     // Exit sensor

// ==================== LCD Initialization ====================
LiquidCrystal lcd(LCD_RS, LCD_E, LCD_D4, LCD_D5, LCD_D6, LCD_D7);

// ==================== System Variables ====================
int peopleCount = 0;
int weaponCount = 0;
String systemStatus = "NORMAL";
bool alertActive = false;
String alertType = "";

// IR Sensor variables
int entryCount = 0;
int exitCount = 0;
int lastIR_ENTRY_STATE = HIGH;
int lastIR_EXIT_STATE = HIGH;
unsigned long lastEntryTime = 0;
unsigned long lastExitTime = 0;
unsigned long irDebounce = 500; // 500ms debounce

// Timing variables
unsigned long lastLCDUpdate = 0;
unsigned long lastBuzzerTime = 0;
unsigned long buzzerCooldown = 10000; // 10 seconds cooldown
unsigned long lcdUpdateInterval = 2000; // Update LCD every 2 seconds

// Thresholds
const int CROWD_THRESHOLD = 5; // Alert when more than 5 people
const int HIGH_CROWD_THRESHOLD = 10; // High alert when more than 10 people

// Display mode
int displayMode = 0;
const int MAX_DISPLAY_MODES = 3;

// ==================== Setup ====================
void setup() {
  // Initialize LED FIRST to show Arduino is alive
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);
  pinMode(BUZZER, OUTPUT);
  
  // Initialize IR sensor pins
  pinMode(IR_ENTRY, INPUT);
  pinMode(IR_EXIT, INPUT);
  
  // Turn on green LED immediately
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BUZZER, LOW);
  
  // Small delay before LCD init
  delay(500);
  
  // Initialize LCD
  lcd.begin(16, 2);
  delay(200);
  lcd.clear();
  delay(200);
  
  // Initialize serial communication
  Serial.begin(9600);
  
  // Show startup message
  lcd.setCursor(0, 0);
  lcd.print("Object Detect");
  lcd.setCursor(0, 1);
  lcd.print("System Ready...");
  
  delay(2000);
  lcd.clear();
  
  Serial.println("ARDUINO_READY");
}

// ==================== Main Loop ====================
void loop() {
  // Check for serial commands from Python
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    parseCommand(command);
  }
  
  // Read IR sensors
  readIRSensors();
  
  // Update LCD periodically
  unsigned long currentMillis = millis();
  if (currentMillis - lastLCDUpdate >= lcdUpdateInterval) {
    updateLCD();
    lastLCDUpdate = currentMillis;
  }
  
  // Update LED status
  updateLEDs();
}

// ==================== Command Parser ====================
void parseCommand(String command) {
  // Command format: COMMAND_TYPE:VALUE
  
  if (command.startsWith("PEOPLE:")) {
    // Update people count
    String value = command.substring(7);
    peopleCount = value.toInt();
    Serial.print("Received PEOPLE_COUNT: ");
    Serial.println(peopleCount);
    
  } else if (command.startsWith("WEAPON:")) {
    // Weapon detected
    String value = command.substring(7);
    weaponCount = value.toInt();
    alertType = "WEAPON";
    Serial.println("===========================");
    Serial.println("[ARDUINO] WEAPON COMMAND RECEIVED!");
    Serial.print("[ARDUINO] Weapon count: ");
    Serial.println(weaponCount);
    Serial.println("[ARDUINO] Triggering alert...");
    triggerAlert();
    Serial.println("===========================");
    
  } else if (command.startsWith("STATUS:")) {
    // Update system status
    systemStatus = command.substring(7);
    Serial.print("Received STATUS: ");
    Serial.println(systemStatus);
    
  } else if (command.startsWith("CROWD_ALERT:")) {
    // Crowd threshold exceeded
    String value = command.substring(12);
    int crowdLevel = value.toInt();
    alertType = "CROWD";
    
    if (crowdLevel >= 2) {
      triggerHighAlert();
    } else {
      triggerAlert();
    }
    Serial.print("Received CROWD_ALERT: Level ");
    Serial.println(crowdLevel);
    
  } else if (command == "RESET") {
    // Reset system
    resetSystem();
    Serial.println("System Reset");
    
  } else if (command == "CLEAR_ALERT") {
    // Clear active alert
    clearAlert();
    Serial.println("Alert Cleared");
  }
}

// ==================== IR Sensor Functions ====================
void readIRSensors() {
  unsigned long currentMillis = millis();
  
  // Read Entry Sensor
  int entryState = digitalRead(IR_ENTRY);
  if (entryState == LOW && lastIR_ENTRY_STATE == HIGH && (currentMillis - lastEntryTime > irDebounce)) {
    entryCount++;
    peopleCount++;
    lastEntryTime = currentMillis;
    Serial.print("ENTRY DETECTED - Total: ");
    Serial.println(peopleCount);
  }
  lastIR_ENTRY_STATE = entryState;
  
  // Read Exit Sensor
  int exitState = digitalRead(IR_EXIT);
  if (exitState == LOW && lastIR_EXIT_STATE == HIGH && (currentMillis - lastExitTime > irDebounce)) {
    if (peopleCount > 0) {
      exitCount++;
      peopleCount--;
    }
    lastExitTime = currentMillis;
    Serial.print("EXIT DETECTED - Total: ");
    Serial.println(peopleCount);
  }
  lastIR_EXIT_STATE = exitState;
}

// ==================== Alert Functions ====================
void triggerAlert() {
  alertActive = true;
  unsigned long currentMillis = millis();
  
  // Check buzzer cooldown (10 seconds)
  if (currentMillis - lastBuzzerTime >= buzzerCooldown) {
    // Activate buzzer for 1 second
    digitalWrite(BUZZER, HIGH);
    delay(1000);
    digitalWrite(BUZZER, LOW);
    lastBuzzerTime = currentMillis;
    Serial.println("[ARDUINO] Buzzer triggered");
  } else {
    Serial.println("[ARDUINO] Buzzer on cooldown");
  }
  
  // Flash red LED (will be handled in loop for non-blocking operation)
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);
  Serial.println("[ARDUINO] Red LED ON - Alert triggered");
}

void triggerHighAlert() {
  alertActive = true;
  unsigned long currentMillis = millis();
  
  // Activate buzzer for 2 seconds with pattern
  if (currentMillis - lastBuzzerTime >= buzzerCooldown) {
    for (int i = 0; i < 3; i++) {
      digitalWrite(BUZZER, HIGH);
      delay(300);
      digitalWrite(BUZZER, LOW);
      delay(200);
    }
    lastBuzzerTime = currentMillis;
  }
  
  // Flash red LED
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, HIGH);
}

void clearAlert() {
  alertActive = false;
  alertType = "";
  digitalWrite(GREEN_LED, HIGH);
  digitalWrite(RED_LED, LOW);
  digitalWrite(BUZZER, LOW);
}

void resetSystem() {
  peopleCount = 0;
  weaponCount = 0;
  entryCount = 0;
  exitCount = 0;
  systemStatus = "NORMAL";
  alertActive = false;
  alertType = "";
  displayMode = 0;
  clearAlert();
  lcd.clear();
}

// ==================== LED Control ====================
void updateLEDs() {
  if (!alertActive) {
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(RED_LED, LOW);
  }
}

// ==================== LCD Update ====================
void updateLCD() {
  lcd.clear();
  
  // Cycle through display modes
  switch (displayMode) {
    case 0:
      // Mode 0: People Count & Status
      displayPeopleCount();
      break;
    case 1:
      // Mode 1: System Status & Alerts
      displaySystemStatus();
      break;
    case 2:
      // Mode 2: Full Stats
      displayFullStats();
      break;
  }
  
  // Update display mode for next cycle
  displayMode = (displayMode + 1) % MAX_DISPLAY_MODES;
}

void displayPeopleCount() {
  lcd.setCursor(0, 0);
  lcd.print("In: ");
  lcd.print(peopleCount);
  lcd.print(" E:");
  lcd.print(entryCount);
  
  lcd.setCursor(0, 1);
  lcd.print("Out:");
  lcd.print(exitCount);
  if (alertActive) {
    lcd.print(" ");
    lcd.print(alertType);
  }
}

void displaySystemStatus() {
  lcd.setCursor(0, 0);
  lcd.print("System: ");
  lcd.print(systemStatus);
  
  lcd.setCursor(0, 1);
  lcd.print("Weapons: ");
  lcd.print(weaponCount);
}

void displayFullStats() {
  lcd.setCursor(0, 0);
  lcd.print("P:");
  lcd.print(peopleCount);
  
  if (peopleCount >= HIGH_CROWD_THRESHOLD) {
    lcd.print(" HIGH");
  } else if (peopleCount >= CROWD_THRESHOLD) {
    lcd.print(" CROWD");
  } else {
    lcd.print(" OK");
  }
  
  lcd.setCursor(0, 1);
  lcd.print("E:");
  lcd.print(entryCount);
  lcd.print(" X:");
  lcd.print(exitCount);
}

// ==================== Helper Functions ====================
void printSystemInfo() {
  Serial.println("=== System Info ===");
  Serial.print("People Count: ");
  Serial.println(peopleCount);
  Serial.print("Weapon Count: ");
  Serial.println(weaponCount);
  Serial.print("System Status: ");
  Serial.println(systemStatus);
  Serial.print("Alert Active: ");
  Serial.println(alertActive);
  Serial.println("===================");
}
