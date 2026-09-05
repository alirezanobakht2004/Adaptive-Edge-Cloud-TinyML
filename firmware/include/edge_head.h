#pragma once

#include <stddef.h>
#include <stdint.h>

namespace inference {

constexpr size_t EDGE_HEAD_INPUT_UNITS = 32;
constexpr size_t EDGE_HEAD_CLASS_COUNT = 5;

bool runEdgeHeadWithMask(
    const float embedding[
        EDGE_HEAD_INPUT_UNITS
    ],
    const uint8_t keepMask[
        EDGE_HEAD_INPUT_UNITS
    ],
    float probabilities[
        EDGE_HEAD_CLASS_COUNT
    ]
);

int edgeHeadArgmax(
    const float probabilities[
        EDGE_HEAD_CLASS_COUNT
    ]
);

}  // namespace inference
