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

//#include "main_functions.h"

#include "accelerometer_handler.h"
#include "constants.h"
#include "gesture_predictor.h"
#include "magic_wand_model_data.h"
#include "output_handler.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <Arduino.h>
#include <Sipeed_ST7789.h>
#include <Wire.h>
#include "lcd.h"

#define WIDTH 240
#define HEIGHT 240
#define AXP173_ADDR 0x34

SPIClass spi_(SPI0); // MUST be SPI0 for Maix series on board LCD
Sipeed_ST7789 lcd(WIDTH, HEIGHT, spi_);

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
int input_length;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 60 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void axp173_init() {
  Wire.begin((uint8_t) SDA, (uint8_t) SCL, 400000);
  Wire.beginTransmission(AXP173_ADDR);
  int err = Wire.endTransmission();
  if (err) {
    Serial.printf("Power management ic not found.\n");
    return;
  }
  Serial.printf("AXP173 found.\n");
  // Clear the interrupts
  Wire.beginTransmission(AXP173_ADDR);
  Wire.write(0x46);
  Wire.write(0xFF);
  Wire.endTransmission();
  // set target voltage and current of battery(axp173 datasheet PG.)
  // charge current (default)780mA -> 190mA
  Wire.beginTransmission(AXP173_ADDR);
  Wire.write(0x33);
  Wire.write(0xC1);
  Wire.endTransmission();
  // REG 10H: EXTEN & DC-DC2 control
  Wire.beginTransmission(AXP173_ADDR);
  Wire.write(0x10);
  Wire.endTransmission();
  Wire.requestFrom(AXP173_ADDR, 1, 1);
  int reg = Wire.read();
  Wire.beginTransmission(AXP173_ADDR);
  Wire.write(0x10);
  Wire.write(reg & 0xFC);
  Wire.endTransmission();
}

// The name of this function is important for Arduino compatibility.
void setup() {
  pll_init();
  plic_init();
//  dmac_init();
  uarths_init();
  Serial.begin(115200);
  axp173_init();
  Serial.printf("started.\n");

  lcd.begin(15000000, COLOR_BLACK);
  lcd.setRotation(2); // 
  tft_write_command(INVERSION_DISPALY_ON);

  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
//  static tflite::MicroMutableOpResolver<5> micro_op_resolver;  // NOLINT
  static tflite::MicroMutableOpResolver micro_op_resolver;  // NOLINT
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor.
  model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != 128) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report(
                         "Bad input tensor parameters in model");
    return;
  }

  input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
  }
}

void loop() {
  // Attempt to read new data from the accelerometer.
  bool got_data =
      ReadAccelerometer(error_reporter, model_input->data.f, input_length);
  // If there was no new data, wait until next time.
  if (!got_data) return;

  // Run inference, and report any error.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    error_reporter->Report("Invoke failed on index: %d\n",
                         begin_index);
    return;
  }

  // Analyze the results to obtain a prediction
  int gesture_index = PredictGesture(interpreter->output(0)->data.f);

  // Produce an output
  HandleOutput(error_reporter, gesture_index);
}
