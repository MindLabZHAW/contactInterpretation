/*  send a frame from can bus
    support@longan-labs.cc
    
    CAN Baudrate,
    
    #define CAN_5KBPS           1
    #define CAN_10KBPS          2
    #define CAN_20KBPS          3
    #define CAN_25KBPS          4 
    #define CAN_31K25BPS        5
    #define CAN_33KBPS          6
    #define CAN_40KBPS          7
    #define CAN_50KBPS          8
    #define CAN_80KBPS          9
    #define CAN_83K3BPS         10
    #define CAN_95KBPS          11
    #define CAN_100KBPS         12
    #define CAN_125KBPS         13
    #define CAN_200KBPS         14
    #define CAN_250KBPS         15
    #define CAN_500KBPS         16
    #define CAN_666KBPS         17
    #define CAN_1000KBPS        18
    
    CANBed V1: https://www.longan-labs.cc/1030008.html
    CANBed M0: https://www.longan-labs.cc/1030014.html
    CAN Bus Shield: https://www.longan-labs.cc/1030016.html
    OBD-II CAN Bus GPS Dev Kit: https://www.longan-labs.cc/1030003.html
*/
#include <Arduino.h>
#include "mcp_can.h"
#include <SPI.h>

/* Please modify SPI_CS_PIN to adapt to different baords.

   CANBed V1        - 17
   CANBed M0        - 3
   CAN Bus Shield   - 9
   CANBed 2040      - 9
   CANBed Dual      - 9
   OBD-2G Dev Kit   - 9
   Hud Dev Kit      - 9
*/

#define SPI_CS_PIN  17 
MCP_CAN CAN(SPI_CS_PIN);                                    // Set CS pin

#define digital_pin  12  // the pin number which should be read by digital pin....  we have only 4 ADC channel
void setup()
{
    Serial.begin(115200);
    //while(!Serial);
    
    pinMode(digital_pin,INPUT);
    
    while (CAN_OK != CAN.begin(CAN_1000KBPS)) // init can bus : baudrate = 500k
    {    
        Serial.println("CAN BUS FAIL!");
        delay(100);
    }

    Serial.println("CAN BUS OK!");
    Serial.println("CAN_1000KBPS");
   
}

unsigned int data[5] = {0, 0, 0, 0, 0};
unsigned int data_old[5] = {0, 0, 0, 0, 0};
//unsigned int fingers[5] = {A3, 12, A0, A1, A2};

//%configure the pinout according to your hardware setup

unsigned int fingers[1] = {A2};

int adc_data;
#define alpha  0.7

#define finger_numbers 1
#define show_raw_data 0 // change it to 1 for showing the raw data
#define serial_on 0 // change it to 1 for prining data on serial monitor
#define limit_val 1019

bool contactState = false;

void loop()
{
    unsigned char dataSend2CAN[5] = {0, 0, 0, 0, 0};
    
    for(int i = 0; i<finger_numbers ; i++){
        if(fingers[i] == digital_pin)
          data[i] =  digitalRead(fingers[i]) + limit_val;
        else{
          adc_data = analogRead(fingers[i]);
          data[i] = (1-alpha)* adc_data + alpha * data_old[i];
          data_old[i] = data[i];//adc_data;
        }
        
        if(data[i] > limit_val & show_raw_data ==0){
          if(serial_on ==1){
            Serial.print("A contact is detected in finger number");
            Serial.println(i+1);
          }
          dataSend2CAN[i] = 1;
          contactState = true;
        }
        if(show_raw_data ==1){
          Serial.print(data[i]);
          Serial.print("  ,   ");
          if(i==finger_numbers-1)
            Serial.println( "  ;");
        }
    }
    
    if(contactState){
      if(serial_on ==1){
        Serial.println("Sending data to computer");
      }
      CAN.sendMsgBuf(0x00, 0, 5, dataSend2CAN);
    }
    contactState = false;
    //delay(1000);                       // send data per 1000ms
}

// END FILE
