#include <ArduinoBLE.h>
#define SERVICE_UUID "0x1815" //for automation io devices in general
#define COMMAND_UUID "19B10001-E8F2-537E-4F6C-D104768A1214" //random uuid for command characteristic
#define STATUS_UUID "19B10002-E8F2-537E-4F6C-D104768A1214" //random uuid for status characteristic

const int Stepper1_Dir = 2;
const int Stepper1_Step = 3;
const int Stepper1_Enable = 4;

const int Stepper2_Dir = 5;
const int Stepper2_Step = 6;
const int Stepper2_Enable = 7;

const int Stepper3_Dir = 8;
const int Stepper3_Step = 9;
const int Stepper3_Enable = 10;

const int Stepper4_Dir = 11;
const int Stepper4_Step = 12;
const int Stepper4_Enable = 13;

int Stepper1_StepsPerRevolution = 1800 * 2; // 1800 for 100:1 with half microsteps
int Stepper2_StepsPerRevolution = 1800 * 2; // 1800 for 100:1 with half microsteps
int Stepper3_StepsPerRevolution = 1800 * 2; // 1800 for 100:1 with half microsteps
int Stepper4_StepsPerRevolution = 1800 * 2; // 1800 for 100:1 with half microsteps

int numberOfSteps = 1800;
int pulseWidthMicros = 20;  // microseconds
int millisbetweenSteps = 1; // milliseconds - or try 1000 for slower steps



BLEService motorService(SERVICE_UUID);
BLEStringCharacteristic commandCharacteristic(COMMAND_UUID, BLEWrite, 30);
BLEStringCharacteristic statusCharacteristic(STATUS_UUID, BLERead | BLENotify, 2);


void blePeripheralConnectHandler(BLEDevice central) {
  // central connected event handler
  Serial.print("Connected event, central: ");
  Serial.println(central.address());
}

void blePeripheralDisconnectHandler(BLEDevice central) {
  // central disconnected event handler
  Serial.print("Disconnected event, central: ");
  Serial.println(central.address());
}

void processCommand(String command) {
  command.trim();  // Remove any whitespace
  
  // New format: stepper<num>_<direction>_<steps>
  // Example: stepper_1_1_400 (Stepper 1, forward, 400 steps)
  // Example: stepper_2_0_900 (Stepper 2, reverse, 900 steps)
  if (command.startsWith("stepper")) {
    int firstUnderscore = command.indexOf('_');
    if (firstUnderscore == -1) return;

    int stepperNum = command.substring(7, firstUnderscore).toInt();

    int secondUnderscore = command.indexOf('_', firstUnderscore + 1);
    if (secondUnderscore == -1) return;

    int direction = command.substring(firstUnderscore + 1, secondUnderscore).toInt();
    
    // CHANGED: The rest of the string is now directly converted to an integer for the steps.
    int steps = command.substring(secondUnderscore + 1).toInt();

    moveStepperMotor(stepperNum, direction, steps);
  } else {
    Serial.println("Invalid command format");
  }
}

void moveStepperMotor(int stepperNum, int direction, int steps) {
  
  // Move the appropriate stepper
  if (stepperNum <=4 && stepperNum >= 1) {
    int dirPin, stepPin, enablePin;
    
    // Assign pins based on stepper number
    switch (stepperNum) {
      case 1:
        dirPin = Stepper1_Dir;
        stepPin = Stepper1_Step;
        enablePin = Stepper1_Enable;
        break;
      case 2:
        dirPin = Stepper2_Dir;
        stepPin = Stepper2_Step;
        enablePin = Stepper2_Enable;
        break;
      case 3:
        dirPin = Stepper3_Dir;
        stepPin = Stepper3_Step;
        enablePin = Stepper3_Enable;
        break;
      case 4:
        dirPin = Stepper4_Dir;
        stepPin = Stepper4_Step;
        enablePin = Stepper4_Enable;
        break;
    }
    // Set direction
    digitalWrite(dirPin, direction == 1 ? HIGH : LOW);
    // Enable the stepper
    digitalWrite(enablePin, HIGH);
    // Move the stepper
    for (int i = 0; i < steps; i++) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(pulseWidthMicros);
      digitalWrite(stepPin, LOW);
      delay(millisbetweenSteps);
    }
    // Disable the stepper after moving
    digitalWrite(enablePin, LOW);
    
  } else {
    Serial.println("Invalid stepper number");
  }


}


void setup() { 

  Serial.begin(9600);
  delay(2000);
  pinMode(Stepper1_Dir, OUTPUT);
  pinMode(Stepper1_Step, OUTPUT);
  pinMode(Stepper1_Enable, OUTPUT);

  pinMode(Stepper2_Dir, OUTPUT);
  pinMode(Stepper2_Step, OUTPUT);
  pinMode(Stepper2_Enable, OUTPUT);

  pinMode(Stepper3_Dir, OUTPUT);
  pinMode(Stepper3_Step, OUTPUT);
  pinMode(Stepper3_Enable, OUTPUT);

  pinMode(Stepper4_Dir, OUTPUT);
  pinMode(Stepper4_Step, OUTPUT);
  pinMode(Stepper4_Enable, OUTPUT);


// begin initialization
  if (!BLE.begin()) {
    Serial.println("starting Bluetooth® Low Energy module failed!");
    while (1);
  }
  BLE.setEventHandler(BLEConnected, blePeripheralConnectHandler);
  BLE.setEventHandler(BLEDisconnected, blePeripheralDisconnectHandler);
  
  BLE.setLocalName("StepperMotorBoard1");
  BLE.setAdvertisedService(motorService);
  motorService.addCharacteristic(commandCharacteristic);
  motorService.addCharacteristic(statusCharacteristic);

  BLE.addService(motorService);
  BLE.advertise();
  Serial.println("BLE Motor Control Ready!");
  Serial.println("Bluetooth device active, waiting for connections...");

}

void loop() { 
  // Listen for BLE connections
  BLEDevice client = BLE.central();

  if (client) {
    Serial.print("Connected to central: ");
    Serial.println(client.address());

    // While the central is connected
    while (client.connected()) {
      // If there's data written to the characteristic
      if (commandCharacteristic.written()) {
        // Get the command
        String command = commandCharacteristic.value();
        Serial.print("Received command: ");
        Serial.println(command);
        
        // Process the command
        processCommand(command);
        // Send completion signal
        statusCharacteristic.writeValue("1");
        Serial.println("Sent completion signal: 1");
        
        // Reset status after a short delay
        delay(100);
        statusCharacteristic.writeValue("0");
      }
    }
  
}
}