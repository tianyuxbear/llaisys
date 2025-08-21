import ctypes
import json
import os
import sys
from ctypes import POINTER, byref, c_int, c_long, c_float
from pathlib import Path
from typing import Sequence

import safetensors
import torch

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    Qwen2MetaCStruct,
    Qwen2WeightsCStruct,
    Qwen2WeightsNaming,
)


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # TODO: Implement model constructor
        model_path = Path(model_path)

        # load config
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config

        # load weights
        for file in sorted(model_path.glob("*.safetensors")):
            state_dict = {}
            data_ = safetensors.safe_open(file, framework="pytorch", device="cpu")
            for name_ in data_.keys():
                ## TODO: load the model weights
                state_dict[name_] = data_.get_tensor(name_)

        # create model
        naming = Qwen2WeightsNaming()
        if naming.match(state_dict):
            ndev = 1
            dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
            self.meta = Qwen2MetaCStruct(config)
            print("create meta ok")
            sys.stdout.flush()
            self.weights = Qwen2WeightsCStruct(self.meta, state_dict, naming, ndev)
            print("create weight ok")
            sys.stdout.flush()
            self.model = LIB_LLAISYS.llaisysQwen2ModelCreate(
                byref(self.meta), byref(self.weights), device, dev_ids, ndev
            )
        print("create model ok")
        sys.stdout.flush()

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 128,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # TODO: Implement generate function
        print("==> entry generate", flush=True)

        tokens = list(inputs)
        for _ in range(max_new_tokens):
            last_logits = self.forward(tokens)
            next_token = sample_token(
                last_logits, top_p=top_p, top_k=top_k, temperature=temperature
            )
            tokens.append(next_token)
            if next_token == self.meta.end_token:
                break

        LIB_LLAISYS.llaisysQwen2ModelDestroy(self.model)

        return tokens

    def forward(self, token_ids: Sequence[int]) -> torch.tensor:
        shape = (self.meta.voc,)
        last_logits = torch.zeros(*shape, dtype=torch.float32, device="cpu")
        ntoken = len(token_ids)
        token_id_t = torch.tensor(token_ids, dtype=torch.int64)

        LIB_LLAISYS.llaisysQwen2ModelInfer(
            self.model,
            ctypes.cast(token_id_t.data_ptr(), POINTER(c_long)),
            ntoken,
            ctypes.cast(last_logits.data_ptr(), POINTER(c_float)),
        )

        return last_logits


def sample_token(
    last_logits: torch.Tensor,
    top_k: int = 1,
    top_p: float = 0.8,
    temperature: float = 0.8,
) -> int:
    """
    Samples a token from the given logits using top-k, top-p (nucleus) sampling, and temperature.

    Args:
        last_logits (torch.Tensor): Logits tensor of shape (voc_size,).
        top_p (float): Cumulative probability threshold for nucleus sampling (0.0 to 1.0). If 1.0, no filtering.
        top_k (int): Number of top tokens to consider. If 0, no top-k filtering.
        temperature (float): Temperature to adjust logits (>0). Higher values make sampling more random.

    Returns:
        int: The sampled token ID.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0.")

    # Apply temperature
    logits = last_logits / temperature

    # Compute probabilities
    probs = torch.softmax(logits, dim=-1)

    # Apply top-k filtering if specified
    if top_k > 0:
        top_k_probs, top_k_indices = torch.topk(probs, min(top_k, probs.size(-1)))
        probs = torch.zeros_like(probs).scatter_(-1, top_k_indices, top_k_probs)
        probs = probs / probs.sum()  # Renormalize

    # Apply top-p (nucleus) filtering if top_p < 1.0
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # Find the smallest set where cumulative prob >= top_p, but include at least the top one
        cutoff_index = torch.sum(cumulative_probs < top_p).item() + 1
        filtered_probs = sorted_probs[:cutoff_index]
        filtered_indices = sorted_indices[:cutoff_index]
        probs = torch.zeros_like(probs).scatter_(-1, filtered_indices, filtered_probs)
        probs = probs / probs.sum()  # Renormalize

    # Sample from the filtered distribution
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id
