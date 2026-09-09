#pragma once
#include <cstddef>
#include <cstdint>

namespace policy {
struct MetaResult {
    int action = -1;
    float normalized[6] = {};
    float probabilities[2] = {};
    uint32_t inferenceUs = 0;
};
bool initializeMeta();
bool runMeta(const float raw[6], MetaResult& result);
size_t metaArenaUsed();
}
