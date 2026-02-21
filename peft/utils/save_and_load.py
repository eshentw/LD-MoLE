# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import os
import huggingface_hub
import torch
from huggingface_hub import file_exists, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
import re
from .config import PeftType
from typing import Optional
from .other import (
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    infer_device,
)
from safetensors.torch import load_file as safe_load_file
from transformers.utils import http_user_agent


def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    if state_dict is None:
        state_dict = model.state_dict()
    if model.peft_config.peft_type == PeftType.LORA:
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = model.peft_config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "sparsegen" in k or "lora_route" in k or "global_sparsegen" in k}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k or "sparsegen" in k or "lora_route" in k or "global_sparsegen" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k or "sparsegen" in k or "lora_route" in k or "global_sparsegen" in k:
                    to_return[k] = state_dict[k]
                    if "lora_" in k:
                        bias_name = k.split("lora_")[0] + "bias"
                        if bias_name in state_dict:
                            to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError
    else:
        to_return = {}
        if model.peft_config.inference_mode:
            prompt_embeddings = model.prompt_encoder.embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save()
        to_return["prompt_embeddings"] = prompt_embeddings
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                to_return[key] = value
    return to_return

def set_peft_model_state_dict(model, peft_model_state_dict, lora_id=-1, **kwargs):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
        lora_id (`int`, *optional*, defaults to `-1`):
            The LoRA adapter id. If -1, all LoRA adapters will be loaded. If >= 0, only the LoRA adapter with the
            given id will be loaded. It's useful when loading LoRA adapters with the same name but different ids.
    """
    if lora_id >= 0:
        use_all = kwargs.pop("use_all", False)
        state_dict = {}
        for k, v in peft_model_state_dict.items():
            if f"lora_A" in k or f"lora_B" in k:
                new_key = re.sub(
                    r"lora_([AB])\d*",  # capture the A or B
                    lambda m: f"lora_{m.group(1)}{lora_id}",
                    k,
                )
                state_dict[new_key] = v
            elif use_all:
                state_dict[k] = v
    else:
        state_dict = peft_model_state_dict

    task = kwargs.pop("task", "Unknown")
    for name in state_dict:
        print(f"[{task}] Loading LoRA adapter param: {name}")

    model.load_state_dict(state_dict, strict=False)
    return model

def torch_load(*args, weights_only=True, **kwargs):
    """Call torch.load and handle weights_only.

    Defaults to weights_only=True to anticipate upcoming switch on the PyTorch side.

    """
    return torch.load(*args, weights_only=weights_only, **kwargs)

def load_peft_weights(model_id: str, device: Optional[str] = None, **hf_hub_download_kwargs) -> dict:
    r"""
    A helper method to load the PEFT weights from the HuggingFace Hub or locally

    Args:
        model_id (`str`):
            The local path to the adapter weights or the name of the adapter to load from the HuggingFace Hub.
        device (`str`):
            The device to load the weights onto.
        hf_hub_download_kwargs (`dict`):
            Additional arguments to pass to the `hf_hub_download` method when loading from the HuggingFace Hub.
    """
    path = (
        os.path.join(model_id, hf_hub_download_kwargs["subfolder"])
        if hf_hub_download_kwargs.get("subfolder", None) is not None
        else model_id
    )

    if device is None:
        device = infer_device()

    def get_hub_filename(use_safetensors=True):
        weights_name = SAFETENSORS_WEIGHTS_NAME if use_safetensors else WEIGHTS_NAME
        return (
            os.path.join(hf_hub_download_kwargs["subfolder"], weights_name)
            if hf_hub_download_kwargs.get("subfolder", None) is not None
            else weights_name
        )

    if "user_agent" not in hf_hub_download_kwargs:
        hf_hub_download_kwargs["user_agent"] = http_user_agent()

    if os.path.exists(os.path.join(path, SAFETENSORS_WEIGHTS_NAME)):
        filename = os.path.join(path, SAFETENSORS_WEIGHTS_NAME)
        use_safetensors = True
    elif os.path.exists(os.path.join(path, WEIGHTS_NAME)):
        filename = os.path.join(path, WEIGHTS_NAME)
        use_safetensors = False
    elif huggingface_hub.constants.HF_HUB_OFFLINE:
        # if in offline mode, check if we can find the adapter file locally
        hub_filename = get_hub_filename(use_safetensors=True)
        hf_hub_download_kwargs.pop("local_files_only", None)
        try:
            filename = hf_hub_download(model_id, hub_filename, local_files_only=True, **hf_hub_download_kwargs)
            use_safetensors = True
        except LocalEntryNotFoundError:
            # Could not find safetensors, try pickle. If this also fails, it's fine to let the error be raised here, as
            # it means that the user tried to load a non-cached model in offline mode.
            hub_filename = get_hub_filename(use_safetensors=False)
            filename = hf_hub_download(model_id, hub_filename, local_files_only=True, **hf_hub_download_kwargs)
            use_safetensors = False
    else:
        token = hf_hub_download_kwargs.get("token", None)
        if token is None:
            token = hf_hub_download_kwargs.get("use_auth_token", None)

        hub_filename = get_hub_filename(use_safetensors=True)
        has_remote_safetensors_file = file_exists(
            repo_id=model_id,
            filename=hub_filename,
            revision=hf_hub_download_kwargs.get("revision", None),
            repo_type=hf_hub_download_kwargs.get("repo_type", None),
            token=token,
        )
        use_safetensors = has_remote_safetensors_file

        if has_remote_safetensors_file:
            # Priority 1: load safetensors weights
            filename = hf_hub_download(
                model_id,
                SAFETENSORS_WEIGHTS_NAME,
                **hf_hub_download_kwargs,
            )
        else:
            try:
                filename = hf_hub_download(model_id, WEIGHTS_NAME, **hf_hub_download_kwargs)
            except EntryNotFoundError:
                raise ValueError(
                    f"Can't find weights for {model_id} in {model_id} or in the Hugging Face Hub. "
                    f"Please check that the file {WEIGHTS_NAME} or {SAFETENSORS_WEIGHTS_NAME} is present at {model_id}."
                )

    if use_safetensors:
        if hasattr(torch.backends, "mps") and (device == torch.device("mps")):
            adapters_weights = safe_load_file(filename, device="cpu")
        else:
            adapters_weights = safe_load_file(filename, device=device)
    else:
        adapters_weights = torch_load(filename, map_location=torch.device(device))

    return adapters_weights
