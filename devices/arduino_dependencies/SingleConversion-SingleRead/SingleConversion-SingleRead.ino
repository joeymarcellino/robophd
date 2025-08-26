#include <SPI.h>
#include <AD7193.h>

AD7193 AD7193;
unsigned long ch1Data;
float ch1Voltage;

void setup() {
  
  ///////////////////////////
  // setup Serial and SPI
  ///////////////////////////
  Serial.begin(115200);
  delay(1000);
  AD7193.begin();
  
  ///////////////////////////////////
  // Device setup
  ///////////////////////////////////
  
  //This will append status bits onto end of data - is required for library to work properly
  AD7193.AppendStatusValuetoData();  
  
  // Set the gain of the PGA
  AD7193.SetPGAGain(1);

  // Set the Averaging
  AD7193.SetAveraging(40);

  AD7193.SetSincFilter(1);

  AD7193.SetPsuedoDifferentialInputs(); 

  /////////////////////////////////////
  // Calibrate with given PGA settings - need to recalibrate if PGA setting is changed
  /////////////////////////////////////
  
  AD7193.Calibrate();

  // Debug - Check register map values
  AD7193.ReadRegisterMap();
  
  //////////////////////////////////////
  //Serial.println("\nBegin AD7193 conversion - single conversion (pg 35 of datasheet, figure 25)");
  
  Serial.setTimeout(10);

}

void loop() {
  // put your main code here, to run repeatedly:

  ch1Data = (AD7193.ReadADCChannel(1) >> 8);
  ch1Voltage = AD7193.DataToVoltage(ch1Data);

  Serial.print("ch1Voltage:");
  Serial.println(ch1Voltage);

 
}
