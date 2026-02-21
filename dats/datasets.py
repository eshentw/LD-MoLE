import logging
import torch
import datasets
from torch.utils.data import DataLoader
from functools import partial

from transformers import PretrainedConfig
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(__name__)

# Global constants
CHOICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

# Dataset configurations with specific sample limits
DATASET_CONFIGS = {
    "mmlu": {
        "dataset_name": "cais/mmlu",
        "dataset_config": "all",
        "split": "test",
        "type": "mmlu",
        "test_samples": 2000
        # "test_samples": 100
    },
    "mmlu_pro": {
        "dataset_name": "TIGER-Lab/MMLU-Pro",
        "dataset_config": None,
        "split": "test",
        "type": "mmlu_pro",
        "test_samples": 2000
        # "test_samples": 100
    },
    "arc_easy": {
        "dataset_name": "allenai/ai2_arc",
        "dataset_config": "ARC-Easy",
        "split": "test",
        "type": "arc_e",
        "test_samples": 2376
        # "test_samples": 100
    },
    "arc_challenge": {
        "dataset_name": "allenai/ai2_arc", 
        "dataset_config": "ARC-Challenge",
        "split": "test",
        "type": "arc_c",
        "test_samples": 1172
        # "test_samples": 100
    },
    "swag": {
        "dataset_name": "allenai/swag",
        "dataset_config": "regular",
        "split": "validation",
        "type": "swag",
        "test_samples": 2000
        # "test_samples": 100
    },
    "commonsenseqa": {
        "dataset_name": "tau/commonsense_qa",
        "dataset_config": None,
        "split": "validation", 
        "type": "commonsenseqa",
        "test_samples": 900
        # "test_samples": 100
    },
    "openbookqa": {
        "dataset_name": "allenai/openbookqa",
        "dataset_config": "main",
        "split": "test",
        "type": "openbookqa",
        "test_samples": 500
        # "test_samples": 100
    }
}

NUM_OPTIONS = {
    'mmlu': 4, 
    'mmlu_pro': 10, 
    'arc_e': 4, 
    'arc_c': 4, 
    'openbookqa': 4, 
    'swag': 4, 
    'commonsenseqa': 5
}


# Dataset formatting functions
def format_mmlu(example):
    """Format MMLU dataset examples"""
    def format_subject(subject):
        return subject.replace("_", " ").lower()

    prompt = (
        f'Below is a multiple-choice question about {format_subject(example["subject"])}. '
        f"Please choose the correct answer.\n"
    )
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["choices"])):
        prompt += "\n{}. {}".format(CHOICES[j], example["choices"][j])
    prompt += "\nAnswer: "
    example["source"] = prompt
    example["target"] = CHOICES[example["answer"]]
    example["target_id"] = example["answer"]
    return example

def format_mmlu_pro(example):
    """Format MMLU-Pro dataset examples"""
    def format_subject(subject):
        return subject.replace("_", " ").lower()
    
    prompt = (
        f'Below is a multiple-choice question about {format_subject(example["category"])}.'
        " Please choose the correct answer.\n"
    )
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["options"])):
        prompt += "\n{}. {}".format(CHOICES[j], example["options"][j])
    prompt += "\nAnswer: "
    example["subject"] = example["category"]
    example["source"] = prompt
    example["target"] = example["answer"]
    example["target_id"] = example["answer_index"]
    return example

def format_arc(example):
    """Format ARC dataset examples"""
    prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
    prompt += example["question"] + "\nOptions:"
    for j in range(len(example["choices"]["text"])):
        prompt += "\n{}. {}".format(CHOICES[j], example["choices"]["text"][j])
    prompt += "\nAnswer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = example["choices"]["label"].index(example["answerKey"])
    return example

def format_swag(example):
    """Format SWAG dataset examples"""
    options = ["A", "B", "C", "D"]
    prompt = "Choose the sentence that best completes the start phrase below:\n"
    prompt += "start phrase: " + example["startphrase"] + "\nOptions:\n"
    prompt += "A. " + example["ending0"] + "\n"
    prompt += "B. " + example["ending1"] + "\n"
    prompt += "C. " + example["ending2"] + "\n"
    prompt += "D. " + example["ending3"] + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = options[example["label"]]
    example["target_id"] = CHOICES.index(example["target"])
    return example

def format_commonsenseqa(example):
    """Format CommonsenseQA dataset examples"""
    prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
    prompt += example["question"] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example["choices"]["text"]):
        prompt += CHOICES[i] + ". " + choice + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = CHOICES.index(example["answerKey"])
    return example

def format_openbookqa(example):
    """Format OpenBookQA dataset examples"""
    prompt = "Below is a multiple-choice question. Please choose the correct answer.\n"
    prompt += "Question: " + example["question_stem"] + "\n"
    prompt += "Options:\n"
    for i, choice in enumerate(example["choices"]["text"]):
        prompt += CHOICES[i] + ". " + choice + "\n"
    prompt += "Answer: "
    example["source"] = prompt
    example["target"] = example["answerKey"]
    example["target_id"] = CHOICES.index(example["answerKey"])
    return example

def remove_column(dataset):
    """Remove unnecessary columns from dataset"""
    save_column = ["source", "target", "target_id", "subject"]
    columns_to_remove = []
    for column_name in dataset.column_names:
        if column_name not in save_column:
            columns_to_remove.append(column_name)
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset

def format_source(source, model_type, training=True):
    """Format source text based on model type"""
    if model_type.startswith('qwen'):
        prefix = "<|im_start|>system\nYou are a helpful assistant.\n<|im_end|>\n<|im_start|>user\n"
        source = prefix + source + "\n<|im_end|>\n"
        if training:
            return source
        return source + "<|im_start|>assistant\n"

def format_target(target):
    target = target + '<|endoftext|>'
    return target

def gen_few_shot_prompt(few_shot_df, k):
    """Generate few-shot prompts"""
    prompt = ""
    if few_shot_df is None:
        return prompt
    for i in range(k):
        prompt += format_example(few_shot_df[i])
    return prompt

def format_example(example, include_answer=True):
    """Format a single example"""
    prompt = example["source"]
    if include_answer:
        prompt += "{}\n\n".format(example["target"])
    return prompt

def load_qwen_datasets(cfg):
    """
    Load all Qwen evaluation datasets from HuggingFace
    
    Args:
        batch_size: Batch size for DataLoader
        use_custom_limits: Whether to use the predefined test_samples limits
    
    Returns:
        dict: Dictionary of dataset loaders
    """
    dataset_loaders = []
    
    for dataset_name, dataset_info in DATASET_CONFIGS.items():
        try:
            print(f"Loading {dataset_name}...")
            
            if dataset_info["dataset_config"]:
                dataset = datasets.load_dataset(
                    dataset_info["dataset_name"], 
                    dataset_info["dataset_config"], 
                    split=dataset_info["split"]
                )
            else:
                dataset = datasets.load_dataset(
                    dataset_info["dataset_name"], 
                    split=dataset_info["split"]
                )            
            data_type = dataset_info["type"]
            if data_type == "mmlu":
                dataset = dataset.map(format_mmlu)
            elif data_type == "mmlu_pro":
                dataset = dataset.map(format_mmlu_pro)
            elif data_type in ["arc_e", "arc_c"]:
                dataset = dataset.map(format_arc)
            elif data_type == "swag":
                dataset = dataset.map(format_swag)
            elif data_type == "commonsenseqa":
                dataset = dataset.map(format_commonsenseqa)
            elif data_type == "openbookqa":
                dataset = dataset.map(format_openbookqa)
            else:
                print(f"⚠️  Unknown data type: {data_type}")
                continue
            dataset = remove_column(dataset)
            
            if "test_samples" in dataset_info:
                test_samples = dataset_info["test_samples"]
                if len(dataset) > test_samples:
                    dataset = dataset.select(range(test_samples))
                    print(f"   Limited to {test_samples} samples")

            batch_size = cfg.val_batch_size if hasattr(cfg, 'val_batch_size') else cfg.batch_size
            dataset_loader = DataLoader(dataset, 
                                        batch_size=batch_size,
                                        num_workers=cfg.num_workers,
                                        shuffle=False,
                                        pin_memory=True)
            
            dataset_loaders.append(dataset_loader)
                
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    return dataset_loaders

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def preprocess_glue(cfg, datasets, tokenizer):

    name = cfg.dataset.name
    if name != 'glue':
        raise NotImplementedError(f"Dataset {name} not supported or using the wrong processing function")
    task = cfg.task.name
    # Labels
    if task is not None:
        is_regression = task == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Preprocessing the datasets
    if task is not None:
        sentence1_key, sentence2_key = task_to_keys[task]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # # Padding strategy
    # if cfg.task.get('pad_to_max_length', None):
    #     padding = "max_length"
    # else:
    #     # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    #     padding = False
    padding = "max_length"

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None

    max_seq_length = cfg.task.get('max_seq_length', None)
    if max_seq_length is None:
        max_seq_length = tokenizer.model_max_length

    if max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)
    return sentence1_key, sentence2_key, padding, max_seq_length, label_to_id, num_labels


def preprocess_function_glue(
                        examples,
                        tokenizer,
                        sentence1_key,
                        sentence2_key,
                        padding,
                        max_seq_length,
                        label_to_id
                    ):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        
    return result


def preprocess_function_flan(examples,
                            tokenizer,
                            padding,
                            max_seq_length,
                            ignore_index):
    """
    Alternative preprocessing function using the format_source and format_target functions
    """
    # Qwen3 require specific formatting
    # @TODO: If using a different model, adjust the formatting accordingly
    formatted_sources = [format_source(source) for source in examples['source']]
    formatted_targets = [format_target(target) for target in examples['target']]
    
    tokenized = tokenizer(
        formatted_sources,
        formatted_targets,
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
        return_tensors='pt',
        add_special_tokens=False,
        return_token_type_ids=True,
    )
    input_ids = tokenized["input_ids"]
    token_type_ids = tokenized["token_type_ids"]
    labels = input_ids.clone()
    # Mask out source tokens from loss
    labels[token_type_ids == 0] = ignore_index
    tokenized["labels"] = labels
    return tokenized