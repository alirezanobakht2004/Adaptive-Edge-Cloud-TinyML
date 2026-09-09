#include "meta_learner.h"
#include <Arduino.h>
#include <cmath>
#include <cstring>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "meta_policy_data.h"

namespace policy {
namespace {
alignas(16) uint8_t arena[8192];
tflite::MicroInterpreter* interpreter = nullptr;
}

bool initializeMeta() {
    if (interpreter) return true;
    static tflite::MicroMutableOpResolver<2> resolver;
    static bool registered = false;
    if (!registered) {
        if (resolver.AddFullyConnected() != kTfLiteOk || resolver.AddSoftmax() != kTfLiteOk) return false;
        registered = true;
    }
    const auto* model = tflite::GetModel(meta_policy_data::MODEL);
    if (model->version() != TFLITE_SCHEMA_VERSION) return false;
    static tflite::MicroInterpreter runtime(model, resolver, arena, sizeof(arena));
    if (runtime.AllocateTensors() != kTfLiteOk) return false;
    if (runtime.input(0)->type != kTfLiteFloat32 || runtime.input(0)->bytes != 6 * sizeof(float)
        || runtime.output(0)->type != kTfLiteFloat32 || runtime.output(0)->bytes != 2 * sizeof(float)) return false;
    interpreter = &runtime;
    return true;
}

bool runMeta(const float raw[6], MetaResult& result) {
    if (!raw || !interpreter) return false;
    for (unsigned i = 0; i < 6; ++i) {
        if (!std::isfinite(raw[i])) return false;
        result.normalized[i] = float((double(raw[i]) - meta_policy_data::OFFSET[i]) / meta_policy_data::DIVISOR[i]);
        if (!std::isfinite(result.normalized[i])) return false;
    }
    memcpy(interpreter->input(0)->data.f, result.normalized, sizeof(result.normalized));
    const uint32_t started = micros();
    if (interpreter->Invoke() != kTfLiteOk) return false;
    result.inferenceUs = micros() - started;
    memcpy(result.probabilities, interpreter->output(0)->data.f, sizeof(result.probabilities));
    if (!std::isfinite(result.probabilities[0]) || !std::isfinite(result.probabilities[1])) return false;
    result.action = result.probabilities[1] > result.probabilities[0] ? 1 : 0;
    return true;
}

size_t metaArenaUsed() { return interpreter ? interpreter->arena_used_bytes() : 0; }
}
