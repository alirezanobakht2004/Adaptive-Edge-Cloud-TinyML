#pragma once

#include <stddef.h>
#include <stdint.h>

namespace internal_tensor_vectors {

constexpr char MODEL_SHA256[] = "50221c98a62c546cb37dcf3537433aa2fb9f0050a37aaa833bc4b1c256ac4dd9";

constexpr size_t PROBE_COUNT = 2;
constexpr size_t STAGE_COUNT = 5;
constexpr size_t INPUT_COUNT = 10;

struct StageExpectation {
    const char* name;
    int tensor_index;
    size_t value_count;
    const int8_t* expected;
};

struct ProbeExpectation {
    int validation_index;
    int true_class;
    const int8_t* input;
    const StageExpectation* stages;
};

static const int8_t VECTOR_82_INPUT[INPUT_COUNT] = {
    -2, -8, -38, -10, -8, -10, -4, -22, -21, -13
};

static const int8_t VECTOR_82_FC1[64] = {
    -124, -99, -128, -113, -128, -116, -102, -97, -115, -128, -128, -109, -104, -114, -119, -128, -128, -120, -128, -128, -119, -104, -128, -128, -102, -120, -112, -114, -128, -123, -128, -122, -121, -128, -102, -128, -128, -110, -128, -115, -128, -115, -105, -128, -128, -128, -128, -128, -128, -103, -128, -120, -103, -128, -96, -93, -101, -128, -128, -110, -116, -123, -128, -108
};

static const int8_t VECTOR_82_FC2[48] = {
    -82, -128, -128, -95, -128, -83, -128, -128, -114, -107, -128, -128, -78, -100, -93, -125, -82, -93, -124, -128, -106, -111, -128, -83, -98, -100, -128, -81, -96, -128, -93, -128, -101, -113, -128, -114, -103, -84, -128, -119, -128, -87, -99, -109, -122, -128, -81, -128
};

static const int8_t VECTOR_82_FC3[32] = {
    -108, -73, -69, -114, -118, -128, -128, -128, -128, -128, -128, -128, -128, -112, -128, -121, -126, -128, -87, -116, -98, -77, -128, -113, -108, -115, -128, -128, -57, -90, -117, -121
};

static const int8_t VECTOR_82_FC4[5] = {
    -53, -1, -21, -46, 1
};

static const int8_t VECTOR_82_SOFTMAX[5] = {
    -128, -26, -126, -128, 24
};

static const StageExpectation VECTOR_82_STAGES[STAGE_COUNT] = {
    {"FC1", 9, 64, VECTOR_82_FC1},
    {"FC2", 10, 48, VECTOR_82_FC2},
    {"FC3", 11, 32, VECTOR_82_FC3},
    {"FC4", 12, 5, VECTOR_82_FC4},
    {"SOFTMAX", 13, 5, VECTOR_82_SOFTMAX},
};

static const int8_t VECTOR_119_INPUT[INPUT_COUNT] = {
    25, 78, -8, 21, 20, 54, -2, -20, -18, -5
};

static const int8_t VECTOR_119_FC1[64] = {
    -110, -63, -128, -79, -128, -71, -58, -37, -112, -128, -88, -128, -39, -121, -128, -128, -128, -49, -128, -128, -128, -128, -92, -110, -128, -90, -97, -78, -128, -128, -60, -118, -96, -118, -125, -128, -80, -109, -128, -128, -128, -89, -128, -128, -96, -128, -128, -61, -128, -128, -123, -117, -5, -128, -128, -95, -110, -128, -128, -66, -87, -128, -128, -104
};

static const int8_t VECTOR_119_FC2[48] = {
    -37, -128, -128, -75, -112, -118, -128, -112, -90, -110, -128, -120, -54, -90, -38, -128, -79, -121, -101, -106, -45, -117, -128, -42, -60, -24, -87, -11, -93, -119, 25, -121, -87, -104, -128, -52, -90, -106, -128, -128, -128, -60, 0, -27, -85, -126, -128, -128
};

static const int8_t VECTOR_119_FC3[32] = {
    -66, -14, -69, -104, -86, -128, -128, -128, -101, -128, -128, -111, -121, -89, -128, -97, -128, -128, -28, -109, -83, -8, -128, -81, -72, -89, -128, -112, -41, -113, -108, -50
};

static const int8_t VECTOR_119_FC4[5] = {
    -76, -10, 15, -50, 33
};

static const int8_t VECTOR_119_SOFTMAX[5] = {
    -128, -128, -121, -128, 121
};

static const StageExpectation VECTOR_119_STAGES[STAGE_COUNT] = {
    {"FC1", 9, 64, VECTOR_119_FC1},
    {"FC2", 10, 48, VECTOR_119_FC2},
    {"FC3", 11, 32, VECTOR_119_FC3},
    {"FC4", 12, 5, VECTOR_119_FC4},
    {"SOFTMAX", 13, 5, VECTOR_119_SOFTMAX},
};

static const ProbeExpectation PROBES[PROBE_COUNT] = {
    {82, 4, VECTOR_82_INPUT, VECTOR_82_STAGES},
    {119, 4, VECTOR_119_INPUT, VECTOR_119_STAGES},
};

}  // namespace internal_tensor_vectors
