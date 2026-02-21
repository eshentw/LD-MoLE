import os
from datasets import load_dataset, load_from_disk
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer
from .datasets import *
from .mmlu_pro import MMLUProDataset
from .arc import ARCDataset
from .commonsenseqa import CSQADataset
from .openbookqa import OpenBookQADataset
from .swag import SWAGDataset
from .hellaswag import HellaSWAGDataset

from .mrpc import MRPCDataset
from .cola import COLADataset
from .mnli import MNLIDataset
from .rte import RTEDataset
from .qqp import QQPDataset
from .sst2 import SST2Dataset
from .stsb import STSBDataset

from torch.utils.data import DataLoader
from functools import partial


def collate_fn_flan(batch):
    """
    Collate function for FLAN models.
    Converts lists of input_ids, attention_mask, etc. from datasets.map into stacked tensors.
    Assumes input is already padded.
    """
    # batch is a list of dicts with keys: input_ids, attention_mask, labels, token_type_ids (optional)
    keys = batch[0].keys()
    batch_dict = {
        "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in batch]),
        "attention_mask": torch.stack([torch.tensor(example["attention_mask"]) for example in batch]),
    }
    if "labels" in keys:
        batch_dict["labels"] = torch.stack([torch.tensor(example["labels"]) for example in batch])
    if "token_type_ids" in keys:
        batch_dict["token_type_ids"] = torch.stack([torch.tensor(example["token_type_ids"]) for example in batch])
    return batch_dict


def collate_fn_pt(cfg, tokenizer, training=True):
    def collate_fn(batch, tokenizer, max_seq_length, padding, ignore_index):
        '''
        Post-Tokenization collate function for FLAN models.
        Pads sequences to the specified max_seq_length and formats them.
        '''
        sources = [x['source']for x in batch]
        targets = [x['target'] for x in batch]

        if training:
            tokenized = tokenizer(
                sources,
                targets,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_token_type_ids=True,
                return_tensors='pt',
                add_special_tokens=False
            )
            labels = tokenized.input_ids.clone()
            labels[tokenized.token_type_ids == 0] = ignore_index
            tokenized['labels'] = labels[:, 1:]
        else:
            tokenized = tokenizer(
                sources,
                padding='longest',
                truncation=True,
                max_length=max_seq_length,
                return_tensors='pt',
                add_special_tokens=False
            )
            tokenized['labels'] = torch.stack([x['labels'] for x in batch], dim=0)
        
        if 'subject' in batch[0]:
            tokenized['subject'] = [x['subject'] for x in batch]
        return tokenized

    return partial(collate_fn, 
                   tokenizer=tokenizer,
                   max_seq_length=cfg.task.max_seq_length,
                   padding=cfg.task.get('padding', False),
                   ignore_index=cfg.dataset.get('ignore_index', -100))


def collate_fn_glue(cfg, tokenizer):
    def collate_fn(batch, tokenizer, max_seq_length, padding):
        sources = [x['source'] for x in batch]
        tokenized = tokenizer(
            sources,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            return_token_type_ids=True,
            return_tensors='pt',
            add_special_tokens=False
        )
        tokenized['labels'] = torch.stack([x['labels'] for x in batch], dim=0)
        return tokenized
    return partial(collate_fn, 
                   tokenizer=tokenizer,
                   max_seq_length=cfg.task.max_seq_length,
                   padding=cfg.task.get('padding', False))


def collate_fn(cfg, training, tokenizer):
    dataset = cfg.dataset.name
    if training:
        if dataset == 'flanv2':
            return collate_fn_flan
        elif dataset == 'glue':
            return collate_fn_glue(cfg, tokenizer)
        else:
            return collate_fn_pt(cfg, tokenizer, training)
    else:
        if dataset == 'glue':
            return collate_fn_glue(cfg, tokenizer)
        else:
            return collate_fn_pt(cfg, tokenizer, training)


def get_tokenizer(cfg, cache_dir=None):
    model_name = cfg.model.get('model_name', None)
    if model_name:
        tokenizer = AutoTokenizer.from_pretrained(
                                                model_name, 
                                                cache_dir=cache_dir if cache_dir is not None else None,
                                                use_fast=True,
                                                revision='main'
                                                )
    else:
        raise NotImplementedError("Model name not provided")

    # Tokenizer configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For GLUE classification tasks, use right padding; for generative tasks, use left padding
    dataset_name = cfg.dataset.get('name', '')
    if dataset_name == 'glue':
        tokenizer.padding_side = 'right'  # Better for classification tasks
    else:
        if tokenizer.padding_side == 'right':
            tokenizer.padding_side = 'left'  # Keep left padding for generative tasks
    
    return tokenizer

  
def build_dataset(cfg):
    meta_dataloader = {}
    name = cfg.dataset.name
    task = cfg.task.name
    cache_dir = os.path.join(cfg.dataset.root_dir, name, task)
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    tokenizer = get_tokenizer(cfg, cache_dir=cache_dir)
    if name == "mmlu_pro":
        tokenized_datasets = {
            'train': MMLUProDataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': MMLUProDataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name in ['arc_e', 'arc_c']:
        tokenized_datasets = {
            'train': ARCDataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': ARCDataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name == "commonsenseqa":
        tokenized_datasets = {
            'train': CSQADataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': CSQADataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name == "openbookqa":
        tokenized_datasets = {
            'train': OpenBookQADataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': OpenBookQADataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name == "swag":
        tokenized_datasets = {
            'train': SWAGDataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': SWAGDataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name == "hellaswag":
        tokenized_datasets = {
            'train': HellaSWAGDataset(cfg, image_set='train', tokenizer=tokenizer),
            'validation': HellaSWAGDataset(cfg, image_set='val', tokenizer=tokenizer),
            'test': None
        }
    elif name == 'glue':
        if task == 'mrpc':
            tokenized_datasets = {
                'train': MRPCDataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': MRPCDataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'cola':
            tokenized_datasets = {
                'train': COLADataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': COLADataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'rte':
            tokenized_datasets = {
                'train': RTEDataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': RTEDataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'mnli':
            tokenized_datasets = {
                'train': MNLIDataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': MNLIDataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'qqp':
            tokenized_datasets = {
                'train': QQPDataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': QQPDataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'sst2':
            tokenized_datasets = {
                'train': SST2Dataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': SST2Dataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
        elif task == 'stsb':
            tokenized_datasets = {
                'train': STSBDataset(cfg, image_set='train', tokenizer=tokenizer),
                'validation': STSBDataset(cfg, image_set='val', tokenizer=tokenizer),
                'test': None
            }
    elif name == "flanv2":
        tokenized_datasets_path = cfg.dataset.get('tokenized_datasets_path', None)
        if tokenized_datasets_path:
            tokenized_datasets = {
                'train': load_from_disk(tokenized_datasets_path),
                'validation': None,
                'test': None
            }
        else:
            datasets = load_dataset(task, cache_dir=cache_dir)
            datasets = ARCDataset.split_dataset(
                            datasets, 
                            train_ratio=1.0, 
                            val_ratio=0.0, 
                            test_ratio=0.0,
                            seed=cfg.dataset.get('seed', 42)
                        )
            columns_to_remove = datasets['train'].column_names
            tokenized_datasets = {
                split: datasets[split].map(
                    lambda x: preprocess_function_flan(
                            x, 
                            tokenizer, 
                            cfg.task.get('padding', False), 
                            cfg.task.get('max_seq_length', 1024), 
                            cfg.dataset.get('ignore_index', -100)
                        ),
                    batched=True,
                    load_from_cache_file=not cfg.dataset.get('overwrite_cache', False),
                ) if datasets[split] is not None else None for split in datasets.keys()
            }
            # remove columns that are not needed
            for split in tokenized_datasets.keys():
                if tokenized_datasets[split] is not None:
                    tokenized_datasets[split] = tokenized_datasets[split].remove_columns(columns_to_remove)
            # save tokenized datasets
            for split in tokenized_datasets.keys():
                if tokenized_datasets[split]:
                    tokenized_datasets[split].save_to_disk(os.path.join(cache_dir, split))

    num_labels = cfg.task.get('num_labels', None)
    train_data_collator = collate_fn(cfg, training=True, tokenizer=tokenizer)
    val_data_collator = collate_fn(cfg, training=False, tokenizer=tokenizer)
    train_dataset = tokenized_datasets['train']
    val_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"]

    if cfg.task.get('max_train_samples', None):
        train_dataset = train_dataset.shuffle(seed=cfg.dataset.seed).select(range(cfg.task.max_train_samples))
    if cfg.task.get('max_val_samples', None):
        val_dataset = val_dataset.select(range(cfg.task.max_val_samples))
    if cfg.task.get('max_test_samples', None):
        test_dataset = test_dataset.select(range(cfg.task.max_test_samples))

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, 
        collate_fn=train_data_collator,
        pin_memory=True,
        shuffle=True
        )
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=cfg.val_batch_size, 
            num_workers=cfg.num_workers, 
            collate_fn=val_data_collator,
            pin_memory=True
            )
    elif name == "flanv2":
        val_loader = load_qwen_datasets(cfg)
    else:
        val_loader = None

    if test_dataset:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=cfg.test_batch_size, 
            num_workers=cfg.num_workers, 
            collate_fn=val_data_collator,
            pin_memory=True
            )
    else:
        test_loader = None

    meta_dataloader.update({
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_labels': num_labels,
        'tokenizer': tokenizer,
    })
    if len(meta_dataloader) < 1:
        raise NotImplementedError(f"Dataset {name} not supported")
    return meta_dataloader