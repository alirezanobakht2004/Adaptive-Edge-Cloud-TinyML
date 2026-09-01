#include <Arduino.h>

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println();
    Serial.println("Adaptive Edge-Cloud TinyML");
    Serial.println("Phase 1 bring-up firmware");
}

void loop() {
    Serial.println("ESP32-S3 alive");
    delay(1000);
}