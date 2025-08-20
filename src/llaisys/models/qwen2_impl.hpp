#ifndef __QWEN2_IMPL_H__
#define __QWEN2_IMPL_H__

#include "llaisys.h"
#include "llaisys/models/qwen2.h"
#include <vector>

struct LlaisysQwen2Model {
    struct LlaisysQwen2Meta meta;
    struct LlaisysQwen2Weights weights;
    llaisysDeviceType_t device;
    std::vector<int> device_ids;

    LlaisysQwen2Model(const LlaisysQwen2Meta *meta_, const LlaisysQwen2Weights *weights_, llaisysDeviceType_t device_, const std::vector<int> device_ids_);
};

#endif