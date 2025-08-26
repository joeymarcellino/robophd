#include <SPI.h>
#include <AD7193.h>

AD7193 AD7193;
unsigned long ch1Data;
unsigned long ch2Data;
float ch1Voltage;
float ch2Voltage;
float TwoChannelRatio;

float ch1Scaling = 49.0;
float ch1Offset = 0.002;
float ch2Scaling = 49.0;
float ch2Offset = 0.002;
unsigned long Channels;
unsigned long buffer[4] = {0,0,0,0};
unsigned long data;
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
  AD7193.SetChannels(0x1);
  delay(10);
  AD7193.Calibrate();
  AD7193.SetChannels(0x2);
  delay(10);
  AD7193.Calibrate();

  // Debug - Check register map values
  AD7193.ReadRegisterMap();
  
  //////////////////////////////////////
  //Serial.println("\nBegin AD7193 conversion - single conversion (pg 35 of datasheet, figure 25)");
  
  Serial.setTimeout(10);
  AD7193.SetChannels(0x3);
  delay(10);
  AD7193.SetContinousConversionMode();

  ch1Voltage = 1.;
  ch2Voltage = 0.;
  TwoChannelRatio = 0.;
}


// Dev Notes: 
// If we want to enable on-the-fly setting changes, 
// such as changing averaging time, we can use the serial port 
// and get the Arduino to constantly read within the loop on top of 
// writing the voltage data
// If we change any parameters of the ADC we must RECALIBRATE both chs

// Sometimes the board does not initialise properly: use the SingleConversion 
// script to run it, somehow this resets something and allows continuous conversion again. 

void loop() {
  // AD7193.SetChannels(0x3);
  // delay(10);
  // AD7193.SetContinousConversionMode();
  //delay(10);

  // while(1) {
    // Wait for ADC conversion
    AD7193.WaitForADC();

    // Querying Single Read value
    //SPI.transfer(0x58);

    // Response of chips
    //SPI.transfer(buffer, 4);
    data = AD7193.ReadADCData();
    Channels = data & 0x0000000F;

    // Serial Print
    if (Channels == 0)
    {   
    ch1Voltage = AD7193.DataToVoltage(data >> 8)*1000;
    ch1Voltage = (ch1Voltage - ch1Offset) / ch1Scaling;
    Serial.print("\nCH1Voltage:");
    Serial.print(ch1Voltage, 6);
    //Serial.print(",");
    } 
    else if (Channels == 1) 
    {
      ch2Voltage = AD7193.DataToVoltage(data >> 8)*1000;
      ch2Voltage = (ch2Voltage - ch2Offset) / ch2Scaling;
      Serial.print("\nCH2Voltage:");
      Serial.print(ch2Voltage, 6);
      //Serial.print(",");
      TwoChannelRatio = ch2Voltage/ch1Voltage;
      Serial.print("\nTwoChannelRatio:");
      Serial.print(TwoChannelRatio, 6);
    }
    else {Serial.print("Error in statue register");}
  // }
}
