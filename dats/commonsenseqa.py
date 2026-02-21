import torch
from .base import BaseDataset


class CSQADataset(BaseDataset):
    CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
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
        """Format CSQA dataset examples"""
        prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
        prompt += example["question"] + "\n"
        prompt += "Options:\n"
        for i, choice in enumerate(example["choices"]["text"]):
            prompt += self.CHOICES[i] + ". " + choice + "\n"
        prompt += "\nAnswer: "
        example["source"] = prompt
        example["target"] = example["answerKey"]
        example["labels"] = self.CHOICES.index(example["answerKey"]) if example["answerKey"] in self.CHOICES else -1
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