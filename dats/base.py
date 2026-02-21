import os
from datasets import load_dataset


class BaseDataset:
    def __init__(self, cfg, image_set, tokenizer=None):
        self.image_set = image_set
        self.tokenizer = tokenizer
        self.root_path = cfg.root_dir
        self.model = cfg.model.name
        self.dataset = cfg.dataset.name
        self.task = cfg.task.name
        self.seed = cfg.dataset.get('seed', 42)
        self.dataset_config = cfg.task.get('dataset_config', None)
        self.padding = cfg.task.get('padding', False)
        self.max_seq_length = cfg.task.get('max_seq_length', 1024)
        self.ignore_index = cfg.task.get('ignore_index', -100)

        self.db = self._get_db()

    def _load_data(self, cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
        if self.dataset == 'glue':
            return load_dataset(
                        self.dataset,
                        self.task,
                        cache_dir=cache_dir
                        )
        else:
            return load_dataset(
                            self.task,
                            self.dataset_config,
                            cache_dir=cache_dir
                            )
    
    def __len__(self):
        return len(self.db)

    def format_source(self, source, model_type):
        """Format source text based on model type"""
        # Next token prediction tasks
        if self.dataset in ['mmlu_pro', 'mmlu', 'arc_e', 'arc_c', 'swag', 'commonsenseqa', 'openbookqa', 'hellaswag']:
            if model_type.startswith('qwen'):
                start_token = '<|im_start|>'
                end_token = '<|im_end|>'
            elif model_type.startswith('llama3'):
                start_token = "<|begin_of_text|>"
                end_token = "<|end_of_text|>"
            else:
                start_token = self.tokenizer.bos_token
                end_token = self.tokenizer.eos_token
            # prefix = f"{start_token}system\nYou are a helpful assistant.\n{end_token}\n{start_token}user\n"
            prefix = start_token
            source = prefix + source
            # source += f"{start_token}assistant\n"
        # Sequence classification tasks
        elif self.task in ['cola', 'mrpc', 'rte', 'sst2', 'stsb', 'qnli', 'qqp']:
            if model_type.startswith('qwen'):
                cls_token = ""
                sep_token = '<|endoftext|>'
                end_token = '<|im_end|>'
            elif model_type.startswith('llama3'):
                cls_token = ""
                sep_token = end_token = "<|end_of_text|>"
            elif model_type.startswith('llama2'):
                cls_token = ""
                sep_token = ""
                end_token = self.tokenizer.eos_token
            else:
                cls_token = self.tokenizer.cls_token
                sep_token = self.tokenizer.sep_token
                end_token = self.tokenizer.eos_token
            if isinstance(source, tuple) or isinstance(source, list):
                source = cls_token + source[0] + sep_token + source[1] + end_token
            elif isinstance(source, str):
                source = cls_token + source + end_token
        return source

    def format_target(self, target, model_type):
        if model_type.startswith('qwen'):
            target = target + '<|im_end|>'
        else:
            target = target + self.tokenizer.eos_token
        return target
    
    @staticmethod
    def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        """
        Split a dataset into train, validation, and test sets.
        
        Args:
            dataset: The dataset to split (expects a Hugging Face Dataset object or DatasetDict)
            train_ratio (float): Proportion for training set (default: 0.8)
            val_ratio (float): Proportion for validation set (default: 0.1)
            test_ratio (float): Proportion for test set (default: 0.1)
            seed (int): Random seed for reproducibility (default: 42)
        
        Returns:
            dict: Dictionary containing 'train', 'val', and 'test' datasets
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
        
        if hasattr(dataset, 'keys') and 'train' in dataset.keys():
            dataset = dataset['train']
        
        if val_ratio == 0.0 and test_ratio == 0.0:
            return {'train': dataset, 'validation': None, 'test': None}
        
        train_val_split = dataset.train_test_split(
            test_size=val_ratio + test_ratio, 
            seed=seed
        )
        
        if test_ratio == 0.0:
            return {
                'train': train_val_split['train'],
                'validation': train_val_split['test'],
                'test': None
            }

        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_test_split = train_val_split['test'].train_test_split(
            test_size=1 - val_test_ratio,
            seed=seed
        )
        
        split_datasets = {
            'train': train_val_split['train'],
            'validation': val_test_split['train'],
            'test': val_test_split['test']
        }
        
        return split_datasets
    
    def __getitem__(self, idx):
        raise NotImplementedError("This method should be implemented by subclasses")

    def _get_db(self):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def _map(self, func):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def _format_input(self, examples):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    def _remove_columns(self, data):
        raise NotImplementedError("This method should be implemented by subclasses")