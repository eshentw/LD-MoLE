import torch
from .base import BaseDataset


class COLADataset(BaseDataset):
    keys="sentence"
    def __init__(self, cfg, image_set, tokenizer):
        super().__init__(cfg, image_set, tokenizer)

    def _get_db(self):
        data = self._load_data(self.root_path)
        data = self._map(data)
        if self.image_set == "train":
            data = data['train']
        elif self.image_set == "val":
            data = data['validation']
        elif self.image_set == "test":
            data = data['test']
        else:
            raise ValueError(f"Unknown image_set: {self.image_set}")
        data = self._remove_column(data)
        return data

    def __getitem__(self, idx):
        example = self.db[idx]
        source = example['source']
        source = self.format_source(source, self.model)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        return {
            "source": source,
            "labels": labels,
        }

    def _map(self, data):
        return data.map(self._format_input)
    
    def _format_input(self, example):
        """Format COLA dataset examples"""
        example["source"] = example[self.keys]
        example["labels"] = example["label"]
        return example
    
    def _remove_column(self, data):
        """Remove unnecessary columns from dataset"""
        columns_to_remove = ["idx", self.keys]
        data = data.remove_columns(columns_to_remove)
        return data