// Isolated control harness around the actual production sensor/main pipeline.
// No test command or action override is included in the normal production image.
#include <Arduino.h>
#include <unity.h>
#include "network/wifi_manager.h"
#undef PIO_UNIT_TESTING
#define setup productionSetup
#define loop productionLoop
#include "../../src/main.cpp"
#undef setup
#undef loop
#define PIO_UNIT_TESTING 1

TaskHandle_t disconnectTaskHandle = nullptr;
volatile bool radioDisabled = false;
void disconnectTask(void*) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    network::disconnectWifi(); // Actual ESP32 radio disconnect, off the sampling core.
    radioDisabled = true;
    Serial.println("M10_REAL_WIFI_DISABLED controlled_trigger=true");
    vTaskDelete(nullptr);
}
void testProductionContinuesWithoutWifi() {
    TEST_ASSERT_EQUAL(pdPASS, xTaskCreatePinnedToCore(disconnectTask,"m10-radio-control",4096,
        nullptr,1,&disconnectTaskHandle,0));
    bool triggered=false;
    const uint32_t started=millis();
    while (uint32_t(millis()-started)<30000) {
        productionLoop();
        if (!triggered && windowCount>=10 && network::isWifiConnected()) {
            triggered=true; xTaskNotifyGive(disconnectTaskHandle);
        }
    }
    // Sampling capture ends here; allow the final queued results to be logged.
    delay(3500);
    Serial.printf("M10_LIVE_SUMMARY windows=%lu local_calls=%lu read_failures=%lu overruns=%lu radio_disabled=%d wifi_recovered=%d\n",
        (unsigned long)windowCount,(unsigned long)localInferenceCalls,(unsigned long)readFailures,
        (unsigned long)fullPeriodOverruns,int(radioDisabled),int(network::isWifiConnected()));
    TEST_ASSERT_TRUE(triggered); TEST_ASSERT_TRUE(radioDisabled);
    TEST_ASSERT_EQUAL(windowCount,localInferenceCalls);
    TEST_ASSERT_EQUAL(0,readFailures); TEST_ASSERT_EQUAL(0,fullPeriodOverruns);
    TEST_ASSERT_TRUE(network::isWifiConnected());
}
void setUp() {}
void tearDown() {}
void setup() { productionSetup(); UNITY_BEGIN(); RUN_TEST(testProductionContinuesWithoutWifi); UNITY_END(); }
void loop(){delay(1000);}
