#pragma once
#include "meta_learner.h"
#include "uncertainty.h"
#include "failover.h"

namespace policy {
struct DecisionResult {
    MetaResult policy;
    int finalPrediction = -1;
    bool success = false;
    unsigned secondInferenceCount = 0; // Cached-result API never invokes the gesture model.
    unsigned txBytes = 0, rxBytes = 0;
    uint32_t roundTripUs = 0;
    float serverComputeMs = 0;
    int effectiveAction = 0;
    bool failover = false, wifiConnected = false, mqttConnected = false, requestIssued = false;
    FailoverReason failoverReason = FailoverReason::NONE;
    unsigned localInferenceCount = 1;
    float finalConfidence = 0;
    const char* requestStatus = "NOT_ISSUED";
};
bool connectAdaptiveNetwork();
bool processCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          const float state[6], const char* requestId, uint32_t windowId,
                          DecisionResult& result, bool controlled = false, unsigned localInferenceCount = 1);
bool startAdaptiveRuntime();
bool submitCachedDecision(const float features[10], const inference::UncertaintyResult& cached,
                          float localInferenceMs, uint32_t windowId, unsigned localInferenceCount = 1);
}
