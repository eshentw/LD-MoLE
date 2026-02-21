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
from .peft_model import (
    PeftModel,
    PeftModelForCausalLM,
    PeftModelForSeq2SeqLM,
    PeftModelForSequenceClassification,
    PeftModelForTokenClassification,
)
from .tuners import LoraConfig, PrefixTuningConfig, PromptEncoderConfig, PromptTuningConfig
from .utils import PromptLearningConfig, PeftType
import torch
from typing import TYPE_CHECKING, Any



if TYPE_CHECKING:
    from .config import PeftConfig
    from .tuners.tuners_utils import BaseTuner

# these will be filled by the register_peft_method function
PEFT_TYPE_TO_CONFIG_MAPPING: dict[PeftType, type[PeftConfig]] = {}
PEFT_TYPE_TO_TUNER_MAPPING: dict[PeftType, type[BaseTuner]] = {}
PEFT_TYPE_TO_MIXED_MODEL_MAPPING: dict[PeftType, type[BaseTuner]] = {}



MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {
    "SEQ_CLS": PeftModelForSequenceClassification,
    "SEQ_2_SEQ_LM": PeftModelForSeq2SeqLM,
    "CAUSAL_LM": PeftModelForCausalLM, # here
    "TOKEN_CLS": PeftModelForTokenClassification,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "PROMPT_TUNING": PromptTuningConfig,
    "PREFIX_TUNING": PrefixTuningConfig,
    "P_TUNING": PromptEncoderConfig,
    "LORA": LoraConfig,
}


TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
}


def get_peft_config(config_dict):
    """
    Returns a Peft config object from a dictionary.

    Args:
        config_dict (`Dict[str, Any]`): Dictionary containing the configuration parameters.
    """

    return PEFT_TYPE_TO_CONFIG_MAPPING[config_dict["peft_type"]](**config_dict)


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", token_dim)

    return peft_config


def _prepare_lora_config(peft_config, model_config):
    if peft_config.target_modules is None:
        if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
            raise ValueError("Please specify `target_modules` in `peft_config`")
        peft_config.target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
    # if len(peft_config.target_modules) == 1:
    #     peft_config.fan_in_fan_out = True
    #     peft_config.enable_lora = [True, False, True]
    if peft_config.inference_mode:
        peft_config.merge_weights = True
    return peft_config


def get_peft_model(model, peft_config):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """

    model_config = model.config.to_dict()
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)

    # PeftModelForCausalLM <- PeftModel
    if peft_config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
        peft_config = _prepare_lora_config(peft_config, model_config)
        return PeftModel(model, peft_config)
    if not isinstance(peft_config, PromptLearningConfig): # -------------> here
        peft_config = _prepare_lora_config(peft_config, model_config)
    else:
        peft_config = _prepare_prompt_learning_config(peft_config, model_config)


    return MODEL_TYPE_TO_PEFT_MODEL_MAPPING[peft_config.task_type](model, peft_config)

def inject_adapter_in_model(
    peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = "default", low_cpu_mem_usage: bool = False
) -> torch.nn.Module:
    r"""
    A simple API to create and inject adapter in-place into a model. Currently the API does not support prompt learning
    methods and adaption prompt. Make sure to have the correct `target_names` set in the `peft_config` object. The API
    calls `get_peft_model` under the hood but would be restricted only to non-prompt learning methods.

    Args:
        peft_config (`PeftConfig`):
            Configuration object containing the parameters of the Peft model.
        model (`torch.nn.Module`):
            The input model where the adapter will be injected.
        adapter_name (`str`, `optional`, defaults to `"default"`):
            The name of the adapter to be injected, if not provided, the default adapter name is used ("default").
        low_cpu_mem_usage (`bool`, `optional`, defaults to `False`):
            Create empty adapter weights on meta device. Useful to speed up the loading process.
    """
    if peft_config.is_prompt_learning or peft_config.is_adaption_prompt:
        raise ValueError("`create_and_replace` does not support prompt learning and adaption prompt yet.")

    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING.keys():
        raise ValueError(
            f"`inject_adapter_in_model` does not support {peft_config.peft_type} yet. Please use `get_peft_model`."
        )

    tuner_cls = PEFT_TYPE_TO_TUNER_MAPPING[peft_config.peft_type]

    # By instantiating a peft model we are injecting randomly initialized LoRA layers into the model's modules.
    peft_model = tuner_cls(model, peft_config, adapter_name=adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    return peft_model.model
