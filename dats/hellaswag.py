import torch
from .base import BaseDataset


class HellaSWAGDataset(BaseDataset):
    """
    A dataset class for handling the HellaSWAG dataset.
    It adapts the original SWAG processing logic to fit HellaSWAG's data structure.
    """
    CHOICES = ["A", "B", "C", "D"]
    def __init__(self, cfg, image_set, tokenizer):
        super().__init__(cfg, image_set, tokenizer)

    def _get_db(self):
        """
        Loads, maps, and filters the dataset based on the specified set (train, val, test).
        """
        data = self._load_data(self.root_path)
        if self.image_set == "train":
            data = data['train']
        elif self.image_set == "val":
            data = data['validation']
        elif self.image_set == "test":
            data = data['test']
        else:
            raise ValueError(f"Unknown image_set: {self.image_set}")
        data = self._map(data)
        data = self._remove_column(data)
        return data

    def __getitem__(self, idx):
        """
        Retrieves a single formatted data point from the dataset at the given index.
        """
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
        """Applies the formatting function to the entire dataset."""
        return data.map(self._format_input)

    def _format_input(self, example):
        """
        Formats a HellaSWAG dataset example into a standardized prompt-target structure.

        HellaSWAG provides a context ('ctx') and a list of possible endings ('endings').
        This function constructs a multiple-choice question from these fields.
        """
        prompt = "Choose the most plausible ending for the following context:\n"
        prompt += "Context: " + example["ctx"] + "\nOptions:\n"
        prompt += "A. " + example["endings"][0] + "\n"
        prompt += "B. " + example["endings"][1] + "\n"
        prompt += "C. " + example["endings"][2] + "\n"
        prompt += "D. " + example["endings"][3] + "\n"
        prompt += "Answer: "
        example["source"] = prompt
        example["target"] = self.CHOICES[int(example["label"])]
        example["labels"] = self.CHOICES.index(example["target"])
        example["subject"] = self.task
        return example
    
    def _remove_column(self, data):
        """Removes unnecessary columns from the dataset to save memory."""
        save_column = ["source", "target", "labels", "subject"]
        columns_to_remove = []
        for column_name in data.column_names:
            if column_name not in save_column:
                columns_to_remove.append(column_name)
        data = data.remove_columns(columns_to_remove)
        return data