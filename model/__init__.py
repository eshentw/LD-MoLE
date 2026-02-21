import torch
import os
from torch import nn
from transformers import AutoConfig

from .qwen3 import  Qwen3ForCausalLM, Qwen3ForSequenceClassification
from .llama import LlamaForCausalLM, LlamaForSequenceClassification


def build_model(cfg, num_labels=None):
    model_hf_name = cfg.model.get('model_name', None)
    model_name = cfg.model.get('name', None)
    name = cfg.dataset.get('name', None)
    task = cfg.task.get('name', None)
    model_config = None
    if not num_labels:
        num_labels = cfg.task.get('num_labels', None)
    if model_hf_name:
        cache_dir = os.path.join(os.getcwd(), cfg.dataset.root_dir, model_name, "weights")
        os.makedirs(cache_dir, exist_ok=True)
        # next token prediction task
        if name in ['mmlu_pro', 'arc_e', 'arc_c', 'swag', 'commonsenseqa', 'openbookqa', 'hellaswag']:
            if cfg.model.name.startswith('qwen3'):
                model_config = AutoConfig.from_pretrained(
                                                        model_hf_name,
                                                        cache_dir=cache_dir,
                                                        )

                # Set pad_token_id from config if specified
                if cfg.model.get('pad_token_id') is not None:
                    model_config.pad_token_id = cfg.model.pad_token_id
                    print(f"Setting pad_token_id to: {cfg.model.pad_token_id}")

                bnb_config = None
                if cfg.model.get('apply_quantization', False):
                    quantization_config = cfg.model.get('quantization', None)
                    assert quantization_config is not None, "Quantization config must be provided for quantization"
                if cfg.load_pretrained:
                    print("Loading pretrained model")
                    model = Qwen3ForCausalLM.from_pretrained(
                                                            model_hf_name,
                                                            from_tf=bool(".ckpt" in model_hf_name),
                                                            config=model_config,
                                                            cache_dir=cache_dir,
                                                            quantization_config=bnb_config
                                                            )
                else:
                    model = Qwen3ForCausalLM(
                                            config=model_config, 
                                            quantization_config=bnb_config
                                            )
            elif cfg.model.name.startswith('llama'):
                model_config = AutoConfig.from_pretrained(
                                                        model_hf_name,
                                                        cache_dir=cache_dir,
                                                        )
                # Set pad_token_id from config if specified
                if cfg.model.get('pad_token_id') is not None:
                    model_config.pad_token_id = cfg.model.pad_token_id
                    print(f"Setting pad_token_id to: {cfg.model.pad_token_id}")

                bnb_config = None
                if cfg.model.get('apply_quantization', False):
                    quantization_config = cfg.model.get('quantization', None)
                    assert quantization_config is not None, "Quantization config must be provided for quantization"
                if cfg.load_pretrained:
                    print("Loading pretrained model")
                    model = LlamaForCausalLM.from_pretrained(
                                                            model_hf_name,
                                                            from_tf=bool(".ckpt" in model_hf_name),
                                                            config=model_config,
                                                            cache_dir=cache_dir,
                                                            quantization_config=bnb_config
                                                            )
                else:
                    model = LlamaForCausalLM(
                                            config=model_config, 
                                            quantization_config=bnb_config
                                            )
        # sequence classification tasks
        elif name == 'glue' and task in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
            model_config = AutoConfig.from_pretrained(
                                                    model_hf_name, 
                                                    cache_dir=cache_dir, 
                                                    num_labels=num_labels,
                                                    finetuning_task=task, 
                                                    revision='main'
                                                    )
            # Set pad_token_id from config if specified
            if cfg.model.get('pad_token_id') is not None:
                model_config.pad_token_id = cfg.model.pad_token_id
                print(f"Setting pad_token_id to: {cfg.model.pad_token_id}")
            elif cfg.model.name.startswith('qwen3'):
                if cfg.load_pretrained:
                    print("Loading pretrained model")
                    model = Qwen3ForSequenceClassification.from_pretrained(
                                                                            model_hf_name,
                                                                            from_tf=bool(".ckpt" in model_hf_name),
                                                                            config=model_config,
                                                                            cache_dir=cache_dir
                                                                            )
                else:
                    model = Qwen3ForSequenceClassification(config=model_config) 
            elif cfg.model.name.startswith('llama'):
                if cfg.load_pretrained:
                    print("Loading pretrained model")
                    model = LlamaForSequenceClassification.from_pretrained(
                                                                            model_hf_name,
                                                                            from_tf=bool(".ckpt" in model_hf_name),
                                                                            config=model_config,
                                                                            cache_dir=cache_dir
                                                                            )
                else:
                    model = LlamaForSequenceClassification(config=model_config)
        else:
            raise NotImplementedError("Model name not supported")

    else:
        raise NotImplementedError("Model name not provided")
    
    # if model_config:
    #     print(model_config)

    return model