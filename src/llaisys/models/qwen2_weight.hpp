#ifndef __QWEN2_WEIGHT_H__
#define __QWEN2_WEIGHT_H__

#include "../../tensor/tensor.hpp"
#include "../llaisys_tensor.hpp"
#include "llaisys/models/qwen2.h"

inline LlaisysTensor getInEmbed(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id) {
    std::vector<size_t> shape{meta->voc, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->in_embed);

    return result;
}

inline LlaisysTensor getOutEmbed(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id) {
    std::vector<size_t> shape{meta->voc, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->out_embed);

    return result;
}

inline LlaisysTensor getOutNormW(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id) {
    std::vector<size_t> shape{meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->out_norm_w);

    return result;
}

inline LlaisysTensor getAttnNormW(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_norm_w[layer]);

    return result;
}

inline LlaisysTensor getAttnQ(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_q_w[layer]);

    return result;
}

inline LlaisysTensor getAttnQB(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_q_b[layer]);

    return result;
}

inline LlaisysTensor getAttnK(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    size_t dkvh = meta->dh * meta->nkvh;
    std::vector<size_t> shape{dkvh, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_k_w[layer]);

    return result;
}

inline LlaisysTensor getAttnKB(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    size_t dkvh = meta->dh * meta->nkvh;
    std::vector<size_t> shape{dkvh};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_k_b[layer]);

    return result;
}

inline LlaisysTensor getAttnV(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    size_t dkvh = meta->dh * meta->nkvh;
    std::vector<size_t> shape{dkvh, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_v_w[layer]);

    return result;
}

inline LlaisysTensor getAttnVB(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    size_t dkvh = meta->dh * meta->nkvh;
    std::vector<size_t> shape{dkvh};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_v_b[layer]);

    return result;
}

inline LlaisysTensor getAttnO(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->attn_o_w[layer]);

    return result;
}

inline LlaisysTensor getMLPNorm(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->mlp_norm_w[layer]);

    return result;
}
inline LlaisysTensor getMLPGate(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->di, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->mlp_gate_w[layer]);

    return result;
}

inline LlaisysTensor getMLPUp(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->di, meta->hs};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->mlp_up_w[layer]);

    return result;
}

inline LlaisysTensor getMLPDown(
    const LlaisysQwen2Meta *meta,
    const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int dev_id, size_t layer) {
    std::vector<size_t> shape{meta->hs, meta->di};
    llaisysTensor_t raw_tensor = tensorCreate(shape.data(), shape.size(), meta->dtype, device, dev_id);

    llaisys::tensor_t smart_tensor = raw_tensor->tensor;

    delete raw_tensor;

    LlaisysTensor result;
    result.tensor = smart_tensor;

    tensorLoad(&result, weights->mlp_down_w[layer]);

    return result;
}

#endif