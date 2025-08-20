#ifndef __QWEN2_IMPL_H__
#define __QWEN2_IMPL_H__

#include "../llaisys_tensor.hpp"
#include "llaisys.h"
#include "llaisys/models/qwen2.h"
#include <vector>

struct DeviceResource {
    // Device
    llaisysDeviceType_t device;
    int device_id;

    // Weights
    LlaisysTensor in_embed, out_embed, out_norm_w;
    std::vector<LlaisysTensor> attn_norm_w, attn_q_w, attn_q_b, attn_k_w, attn_k_b, attn_v_w, attn_v_b, attn_o_w, mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;

    // Streams
    llaisysStream_t stream;
};

struct LlaisysQwen2Model {
    struct LlaisysQwen2Meta meta;
    llaisysDeviceType_t device;
    std::vector<int> dev_ids;
    std::vector<DeviceResource> dev_resources;

    LlaisysQwen2Model(const LlaisysQwen2Meta *meta_, const LlaisysQwen2Weights *weights_, llaisysDeviceType_t device_, const std::vector<int> dev_ids_);
};

void createDeviceResource(DeviceResource *rsrc, const LlaisysQwen2Meta *meta,
                          const LlaisysQwen2Weights *weights,
                          llaisysDeviceType_t device, int dev_id);

#endif