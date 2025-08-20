#include "llaisys/models/qwen2.h"
#include "llaisys.h"
#include "llaisys/tensor.h"
#include "qwen2_impl.hpp"
#include <cstddef>
#include <iostream>
#include <vector>

LlaisysQwen2Model::LlaisysQwen2Model(const LlaisysQwen2Meta *meta_, const LlaisysQwen2Weights *weights_, llaisysDeviceType_t device_, const std::vector<int> device_ids_) : meta(*meta_), device(device_), device_ids(device_ids_) {
    std::cout << "from C llaisysQwen2Model Constructor: begin" << std::endl;
    size_t nlayer = meta.nlayer;
    size_t voc = meta.voc;
    size_t hs = meta.hs;
    size_t di = meta.di;
    size_t dh = meta.dh;
    size_t nkvh = meta.nkvh;
    size_t dkv_p = nkvh * dh;

    std::cout << "nlayer: " << nlayer << ", voc: " << voc << ", hs： " << hs << ", di: " << di << ", dh: " << dh << ", nkvh: " << nkvh << ", dkv_p: " << dkv_p << std::endl;

    // in_embed
    std::vector<size_t> shape = {voc, hs};
    weights.in_embed = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
    tensorLoad(weights.in_embed, weights_->in_embed);

    // out_embed
    shape = {voc, hs};
    weights.out_embed = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
    tensorLoad(weights.out_embed, weights_->out_embed);

    // out_norm
    shape = {hs};
    weights.out_norm_w = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
    tensorLoad(weights.out_norm_w, weights_->out_norm_w);

    std::cout << "befor load layer" << std::endl;

    for (size_t i = 0; i < nlayer; ++i) {
        std::cout << "layer " << i << " : begin" << std::endl;
        // attn_norm
        shape = {hs};
        weights.attn_norm_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_norm_w[i], weights_->attn_norm_w[i]);

        // attn_q
        shape = {hs, hs};
        weights.attn_q_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_q_w[i], weights_->attn_q_w[i]);

        // attn_q_b
        shape = {hs};
        weights.attn_q_b[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_q_b[i], weights_->attn_q_b[i]);

        // attn_k
        shape = {dkv_p, hs};
        weights.attn_k_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_k_w[i], weights_->attn_k_w[i]);

        // attn_k_b
        shape = {dkv_p};
        weights.attn_k_b[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_k_b[i], weights_->attn_k_b[i]);

        // attn_v
        shape = {dkv_p, hs};
        weights.attn_v_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_v_w[i], weights_->attn_v_w[i]);

        // attn_v_b
        shape = {dkv_p};
        weights.attn_v_b[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_v_b[i], weights_->attn_v_b[i]);

        // attn_o
        shape = {hs, hs};
        weights.attn_o_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.attn_o_w[i], weights_->attn_o_w[i]);

        // mlp_norm
        shape = {hs};
        weights.mlp_norm_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.mlp_norm_w[i], weights_->mlp_norm_w[i]);

        // mlp_gate
        shape = {di, hs};
        weights.mlp_gate_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.mlp_gate_w[i], weights_->mlp_gate_w[i]);

        // mlp_up
        shape = {di, hs};
        weights.mlp_up_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.mlp_up_w[i], weights_->mlp_up_w[i]);

        // mlp_down
        shape = {hs, di};
        weights.mlp_down_w[i] = tensorCreate(shape.data(), shape.size(), meta.dtype, device, device_ids[0]);
        tensorLoad(weights.mlp_down_w[i], weights_->mlp_down_w[i]);
        std::cout << "layer " << i << " : begin" << std::endl;
    }
    std::cout << "from C llaisysQwen2Model Constructor: end" << std::endl;
}

__C {
    __export struct LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, const LlaisysQwen2Weights *weights, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        std::cout << "from C llaisysQwen2ModelCreate" << std::endl;
        std::vector<int> dev_ids(ndevice);
        std::copy(device_ids, device_ids + ndevice, dev_ids.begin());
        LlaisysQwen2Model *model = new LlaisysQwen2Model(meta, weights, device, dev_ids);
        return model;
    }

    __export void llaisysQwen2ModelDestroy(struct LlaisysQwen2Model * model) {
        delete model;
    }

    __export struct LlaisysQwen2Weights *llaisysQwen2ModelWeights(struct LlaisysQwen2Model * model) {
        return &model->weights;
    }

    __export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken) {
        return 0;
    }
}