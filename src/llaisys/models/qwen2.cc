#include "llaisys/models/qwen2.h"
#include "llaisys.h"
#include "qwen2_impl.hpp"
#include "qwen2_weight.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

void createDeviceResource(DeviceResource *rsrc, const LlaisysQwen2Meta *meta,
                          const LlaisysQwen2Weights *weights,
                          llaisysDeviceType_t device, int dev_id) {
    llaisysSetContextRuntime(device, dev_id);
    const LlaisysRuntimeAPI *api_ = llaisysGetRuntimeAPI(device);
    llaisysStream_t stream = api_->create_stream();

    size_t nlayer = meta->nlayer;

    LlaisysTensor in_embed, out_embed, out_norm_w;
    std::vector<LlaisysTensor> attn_norm_w(nlayer), attn_q_w(nlayer), attn_q_b(nlayer), attn_k_w(nlayer), attn_k_b(nlayer), attn_v_w(nlayer), attn_v_b(nlayer), attn_o_w(nlayer), mlp_norm_w(nlayer), mlp_gate_w(nlayer), mlp_up_w(nlayer), mlp_down_w(nlayer);

    in_embed = getInEmbed(meta, weights, device, dev_id);
    out_embed = getOutEmbed(meta, weights, device, dev_id);
    out_norm_w = getOutNormW(meta, weights, device, dev_id);

    for (size_t layer = 0; layer < nlayer; ++layer) {
        attn_norm_w[layer] = getAttnNormW(meta, weights, device, dev_id, layer);
        attn_q_w[layer] = getAttnQ(meta, weights, device, dev_id, layer);
        attn_q_b[layer] = getAttnQB(meta, weights, device, dev_id, layer);
        attn_k_w[layer] = getAttnK(meta, weights, device, dev_id, layer);
        attn_k_b[layer] = getAttnKB(meta, weights, device, dev_id, layer);
        attn_v_w[layer] = getAttnV(meta, weights, device, dev_id, layer);
        attn_v_b[layer] = getAttnVB(meta, weights, device, dev_id, layer);
        attn_o_w[layer] = getAttnO(meta, weights, device, dev_id, layer);
        mlp_norm_w[layer] = getMLPNorm(meta, weights, device, dev_id, layer);
        mlp_gate_w[layer] = getMLPGate(meta, weights, device, dev_id, layer);
        mlp_up_w[layer] = getMLPUp(meta, weights, device, dev_id, layer);
        mlp_down_w[layer] = getMLPDown(meta, weights, device, dev_id, layer);
    }

    *rsrc = DeviceResource{
        device,
        dev_id,
        in_embed,
        out_embed,
        out_norm_w,
        attn_norm_w,
        attn_q_w,
        attn_q_b,
        attn_k_w,
        attn_k_b,
        attn_v_w,
        attn_v_b,
        attn_o_w,
        mlp_norm_w,
        mlp_gate_w,
        mlp_up_w,
        mlp_down_w,
        stream};
}

LlaisysQwen2Model::LlaisysQwen2Model(const LlaisysQwen2Meta *meta_, const LlaisysQwen2Weights *weights_, llaisysDeviceType_t device_, const std::vector<int> dev_ids_) : meta(*meta_), device(device_), dev_ids(dev_ids_) {
    std::cout << "from C llaisysQwen2Model Constructor: begin" << std::endl;
    size_t ndev = dev_ids.size();
    dev_resources = std::vector<DeviceResource>(ndev);
    for (size_t i = 0; i < ndev; ++i) {
        std::cout << "create device resource: " << i << std::endl;
        createDeviceResource(&dev_resources[i], meta_, weights_, device_, i);
    }
    std::cout << "from C llaisysQwen2Model Constructor: end" << std::endl;
}

__C {
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        std::cout << "from C llaisysQwen2ModelCreate: begin" << std::endl;
        std::vector<int> dev_ids(ndevice);
        std::copy(device_ids, device_ids + ndevice, dev_ids.begin());
        LlaisysQwen2Model *model = new LlaisysQwen2Model(meta, weights, device, dev_ids);
        std::cout << "from C llaisysQwen2ModelCreate: end" << std::endl;
        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model;
    }

    __export void llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, float *last_logits) {
        
    }
}