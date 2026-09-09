#pragma once
#include <cstdint>

namespace policy {
enum class FailoverReason { NONE, WIFI_UNAVAILABLE, MQTT_UNAVAILABLE,
    CLOUD_RESPONSE_TIMEOUT, CLOUD_PUBLISH_FAILED, LOCAL_QUEUE_SAFETY };
inline const char* reasonName(FailoverReason reason) {
    switch (reason) {
        case FailoverReason::WIFI_UNAVAILABLE: return "WIFI_UNAVAILABLE";
        case FailoverReason::MQTT_UNAVAILABLE: return "MQTT_UNAVAILABLE";
        case FailoverReason::CLOUD_RESPONSE_TIMEOUT: return "CLOUD_RESPONSE_TIMEOUT";
        case FailoverReason::CLOUD_PUBLISH_FAILED: return "CLOUD_PUBLISH_FAILED";
        case FailoverReason::LOCAL_QUEUE_SAFETY: return "LOCAL_QUEUE_SAFETY";
        default: return "NONE";
    }
}
// Pure result transition: no gesture-network dependency or inference callback.
struct FailoverState {
    int requestedAction = -1, effectiveAction = 0, cachedClass = -1, finalClass = -1;
    unsigned localInferenceCount = 0;
    bool complete = false, failover = false;
    FailoverReason reason = FailoverReason::NONE;
    bool begin(int requested, bool wifi, bool mqtt, int localClass, unsigned localCalls) {
        *this = FailoverState{};
        requestedAction = requested; cachedClass = localClass; localInferenceCount = localCalls;
        if (localCalls != 1 || localClass < 0 || localClass > 4) return false;
        if (requested == 0) { finalClass = cachedClass; complete = true; return true; }
        if (!wifi) fallback(FailoverReason::WIFI_UNAVAILABLE);
        else if (!mqtt) fallback(FailoverReason::MQTT_UNAVAILABLE);
        return true;
    }
    void fallback(FailoverReason why) {
        effectiveAction = 0; finalClass = cachedClass; failover = true; reason = why; complete = true;
    }
    void cloudSuccess(int predicted) {
        if (!complete && requestedAction == 1 && predicted >= 0 && predicted <= 4) {
            finalClass = predicted; effectiveAction = 1; complete = true;
        }
    }
};
inline bool requestExpired(uint32_t nowUs, uint32_t startedUs, uint32_t timeoutMs) {
    return uint32_t(nowUs - startedUs) >= timeoutMs * 1000U;
}
}
