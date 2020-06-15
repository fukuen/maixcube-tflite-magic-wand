/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "output_handler.h"
#include <Arduino.h>
#include <Sipeed_ST7789.h>
#include "lcd.h"

#define WIDTH 240
#define HEIGHT 240
#define AXP173_ADDR 0x34

//extern SPIClass spi_(SPI0); // MUST be SPI0 for Maix series on board LCD
extern Sipeed_ST7789 lcd;

void DrawWing() {
  int x = 20; //60;
  int y = 20;
  int w = 200;
  int h = 200;
  int t = 20;
  int k;
  k = (w-t)/4;
  lcd.fillTriangle(x, y, x+t, y, x + k, y+h, GREEN);
  lcd.fillTriangle(x+t, y, x+k, y+h, x+k+t, y+h, GREEN);
  lcd.fillTriangle(x+2*k, y, x+2*k+t, y, x+k, y+h, GREEN);
  lcd.fillTriangle(x+2*k+t, y, x+k, y+h, x+k+t, y+h, GREEN);
  x += 2*k;
  lcd.fillTriangle(x, y, x+t, y, x + k, y+h, GREEN);
  lcd.fillTriangle(x+t, y, x+k, y+h, x+k+t, y+h, GREEN);
  lcd.fillTriangle(x+2*k, y, x+2*k+t, y, x+k, y+h, GREEN);
  lcd.fillTriangle(x+2*k+t, y, x+k, y+h, x+k+t, y+h, GREEN);
}

void DrawRing() {
  int x = 20; //60;
  int y = 20;
  int w = 200;
  int h = 200;
  int t = 20;
  lcd.drawCircle(x+w/2, y+h/2, w/2, RED);
  lcd.drawCircle(x+w/2, y+h/2, w/2-t, BLACK);
}

void DrawSlope() {
  int x = 20; //60;
  int y = 20;
  int w = 200;
  int h = 200;
  int t = 20;
  lcd.fillTriangle(x+w-t*1.5, y, x+w, y, x, y+w, BLUE);
  lcd.fillTriangle(x+w, y, x, y+w, x+t*1.5, y+w, BLUE);
  lcd.fillRect(x+t, y+h-t, w-t, t, BLUE);
}

void HandleOutput(tflite::ErrorReporter* error_reporter, int kind) {
  // light (red: wing, blue: ring, green: slope)
  if (kind == 0) {
//    error_reporter->Report(
    Serial.printf(
        "WING:\n\r*         *         *\n\r *       * *       "
        "*\n\r  *     *   *     *\n\r   *   *     *   *\n\r    * *       "
        "* *\n\r     *         *\n\r");
    lcd.fillScreen(BLACK);
    DrawWing();
  } else if (kind == 1) {
//    error_reporter->Report(
    Serial.printf(
        "RING:\n\r          *\n\r       *     *\n\r     *         *\n\r "
        "   *           *\n\r     *         *\n\r       *     *\n\r      "
        "    *\n\r");
    lcd.fillScreen(BLACK);
    DrawRing();
  } else if (kind == 2) {
//    error_reporter->Report(
    Serial.printf(
        "SLOPE:\n\r        *\n\r       *\n\r      *\n\r     *\n\r    "
        "*\n\r   *\n\r  *\n\r * * * * * * * *\n\r");
    lcd.fillScreen(BLACK);
    DrawSlope();
  }
}
