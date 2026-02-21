import torch
from .base import BaseDataset


class SWAGDataset(BaseDataset):
    CHOICES = ["A", "B", "C", "D"]
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
        subject = example['subject']
        target = example['target']
        source = self.format_source(source, self.model)
        target = self.format_target(target, self.model)
        labels = torch.tensor(example['labels'], dtype=torch.long)
        return {
            "source": source,
            "target": target,
            "labels": labels,
            "subject": subject
        }

    def _map(self, data):
        return data.map(self._format_input)

    def _format_input(self, example):
        """Format SWAG dataset examples"""
        prompt = "Choose the sentence that best completes the start phrase below:\n"
        prompt += "start phrase: " + example["startphrase"] + "\nOptions:\n"
        prompt += "A. " + example["ending0"] + "\n"
        prompt += "B. " + example["ending1"] + "\n"
        prompt += "C. " + example["ending2"] + "\n"
        prompt += "D. " + example["ending3"] + "\n"
        prompt += "Answer: "
        example["source"] = prompt
        example["target"] = self.CHOICES[example["label"]]
        example["labels"] = self.CHOICES.index(example["target"])
        example["subject"] = self.task
        return example
    
    def _remove_column(self, data):
        """Remove unnecessary columns from dataset"""
        save_column = ["source", "target", "labels", "subject"]
        columns_to_remove = []
        for column_name in data.column_names:
            if column_name not in save_column:
                columns_to_remove.append(column_name)
        data = data.remove_columns(columns_to_remove)
        return data