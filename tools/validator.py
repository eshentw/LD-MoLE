import torch
import torch.distributed as dist


NUM_OPTIONS = {
    'mmlu_pro': 10, 
    'arc_e': 4, 
    'arc_c': 4, 
    'openbookqa': 4, 
    'swag': 4, 
    'commonsenseqa': 5,
    'hellaswag': 4
}


def build_validator(config):
    dataset = config.dataset.get('name', None)
    if dataset == 'glue':
        return GlueValidator(config)
    elif dataset in ['mmlu_pro', 'arc_e', 'arc_c', 'openbookqa', 'swag', 'commonsenseqa', 'hellaswag']:
        return SingleTaskLMValidator(config)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not supported for validation.")


class BaseValidator:
    def __init__(self, config):
        self.config = config
    
    def init_metrics(self):
        raise NotImplementedError("Not implemented")
    
    def update_metrics(self, preds, labels):
        raise NotImplementedError("Not implemented")
    
    def get_desc(self):
        raise NotImplementedError("Not implemented")
    
    def get_results(self):
        return NotImplementedError("Not implemented")

    def print_metrics(self, metrics):
        """Prints the evaluation metrics in a properly aligned format."""
        # Print the header
        print(self.get_desc())
        format_str = "%22s %11d" + " %11.4f" * len(self.metric_names)
        task_name = self.config.task.get('name', None)
        print(format_str % (
                    task_name,
                    self.num_cnts,
                    *metrics.values())
              )


class GlueValidator(BaseValidator):
    def __init__(self, config):
        self.config = config
        self.metric_names = []
        self.init_metrics()

    def init_metrics(self):
        self.num_cnts = 0
        self.metrics = {}
    
    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        format_str = "%22s %11s" + " %11s" * len(self.metric_names)
        return format_str % (
                    "Task",
                    "Instances",
                    *self.metric_names
                )
        
    def get_metrics(self, preds, labels, rank):
        # Gather predictions from all processes if using distributed training
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            
            gathered_preds = [torch.zeros_like(preds) for _ in range(world_size)]
            gathered_labels = [torch.zeros_like(labels) for _ in range(world_size)]
            dist.all_gather(gathered_preds, preds)
            dist.all_gather(gathered_labels, labels)
            assert isinstance(gathered_preds[0], torch.Tensor), \
                f"Expected gathered_preds[0] to be a Tensor, got {type(gathered_preds[0])}"
            if rank == 0:
                preds = torch.cat(gathered_preds, dim=0)
                labels = torch.cat(gathered_labels, dim=0)

        if rank != 0:
            return None
        task = self.config.task.get('name', None)
        if task in ['cola', 'mrpc', 'sst2', 'rte', 'qqp', 'mnli']:
            preds = preds.argmax(dim=-1) if isinstance(preds, torch.Tensor) else preds
            # For classification tasks, compute accuracy
            if preds.shape != labels.shape:
                raise ValueError(f"Predictions shape {preds.shape} does not match labels shape {labels.shape}")
            correct = (preds == labels).sum().item()
            total = len(labels)
            accuracy = correct / total if total > 0 else 0.0
            results = {'accuracy': accuracy}
        elif task == 'stsb':
            # For regression tasks, compute mean squared error
            preds = preds.squeeze() if isinstance(preds, torch.Tensor) else preds
            mse = ((preds - labels) ** 2).mean().item()
            results = {'mse': mse}
        return results

    def update_metrics(self, results, rank):
        '''Update and accum the metrics with the new predictions and labels.'''
        if rank != 0:
            return
        self.num_cnts += 1
        if not self.metric_names:
            self.metric_names = list(results.keys())

        for k, v in results.items():
            if k not in self.metrics:
                self.metrics[k] = 0
            if isinstance(v, torch.Tensor):
                self.metrics[k] += v.item()
            else:
                self.metrics[k] += v

    def get_results(self):
        metrics_results = {}
        for k, v in self.metrics.items():
            metrics_results[k] = v / self.num_cnts
        self.print_metrics(metrics_results)
        return metrics_results


class SingleTaskLMValidator(BaseValidator):
    def __init__(self, config):
        self.config = config
        self.metric_names = []
        self.dataset_name = config.dataset.get('name', None)
        self.option_list = [
            'A', ' A', 'B', ' B', 
            'C', ' C', 'D', ' D', 
            'E', ' E', 'F', ' F', 
            'G', ' G', 'H', ' H', 
            'I', ' I', 'J', ' J'
        ]
        self.init_metrics()

    def init_metrics(self):
        self.num_cnts = 0
        self.metrics = {}

    def get_desc(self):
        """Returns description of evaluation metrics in string format."""
        format_str = "%22s" + " %11s" * len(self.metric_names)
        return format_str % (
                    "Task",
                    *self.metric_names
                )
        
    def get_metrics(self, preds, labels, batch, rank):
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered_preds = [torch.empty_like(preds) for _ in range(world_size)]
            gathered_labels = [torch.empty_like(labels) for _ in range(world_size)]
            dist.all_gather(gathered_preds, preds)
            dist.all_gather(gathered_labels, labels)
            assert isinstance(gathered_preds[0], torch.Tensor), \
                f"Expected gathered_preds[0] to be a Tensor, got {type(gathered_preds[0])}"
            if rank == 0:
                preds = torch.cat(gathered_preds, dim=0)
                labels = torch.cat(gathered_labels, dim=0)
            all_subjects = []
            if 'subject' in batch:
                gathered_subjects = [None for _ in range(world_size)]
                dist.all_gather_object(gathered_subjects, batch['subject'])
                if rank == 0:
                    for subjects in gathered_subjects:
                        all_subjects.extend(subjects) 
        else:
            all_subjects = batch['subject'] if 'subject' in batch else []

        if rank != 0:
            return None
        results = {}
        data_type = self.dataset_name

        for i in range(len(preds)):
            if data_type in ['mmlu_pro']:
                subject = all_subjects[i]
                task_name = subject.replace('_', ' ').title().lower()
            else:
                task_name = data_type
            if task_name not in results:
                results[task_name] = {"right_count": 0, "all_count": 0, "accuracy": 0.0}
            if preds[i].item() == labels[i].item():
                results[task_name]["right_count"] += 1
            results[task_name]["all_count"] += 1

        for task_name, counts in results.items():
            if counts["all_count"] > 0:
                results[task_name]["accuracy"] = counts["right_count"] / counts["all_count"]
            else:
                results[task_name]["accuracy"] = 0.0
        return results

    def update_metrics(self, results, rank):
        '''Update and accum the metrics with the new predictions and labels.'''
        if rank != 0:
            return
        self.num_cnts += 1
        
        for k, v in results.items():
            if k not in self.metrics:
                self.metrics[k] = {"accuracy": 0.0, "right_count": 0, "all_count": 0}

            self.metrics[k]["right_count"] += v["right_count"]
            self.metrics[k]["all_count"] += v["all_count"]
            self.metrics[k]["accuracy"] = self.metrics[k]["right_count"] / self.metrics[k]["all_count"]

    def get_results(self):
        metrics_results = {}
        
        if self.dataset_name in ['mmlu_pro']:
            for k, v in self.metrics.items():
                met_name = f"{k}_acc"
                metrics_results[met_name] = v["accuracy"]

        total_right = sum(v["right_count"] for v in self.metrics.values())
        total_all = sum(v["all_count"] for v in self.metrics.values())
        mmlu_avg_accuracy = total_right / total_all if total_all > 0 else 0.0
        metrics_results["mean_acc"] = mmlu_avg_accuracy
                
        if not self.metric_names:
            self.metric_names = ["accuracy"]
        self.print_metrics(metrics_results)
        return metrics_results
    
    def print_metrics(self, metrics_results):
        print(self.get_desc())
        format_str = "%22s" + " %11.4f" * len(self.metric_names)
        for subject, accuracy in metrics_results.items():
            print(format_str % (subject, accuracy))
    
    @torch.no_grad()
    def evaluate(
                self,
                res,
                tokenizer,
                ):
        """
        Evaluate model on a single dataset
        
        Args:
            res: Model output logits last token
            tokenizer: Tokenizer for the model
        
        Returns:
            tuple: (accuracy, right_count_by_subject, all_count_by_subject)
        """
        data_type = self.dataset_name
        options = self.option_list[:NUM_OPTIONS[data_type] * 2]
        option_index = tokenizer(options,
                                return_tensors='pt', 
                                add_special_tokens=False).input_ids.squeeze()
        
        with torch.no_grad():
            option_logits = res[:, option_index]  # [batch_size, num_options]
            # Convert to float32 for stable argmax computation
            preds = torch.argmax(option_logits.float(), dim=-1)
            preds = preds // 2
        
        return preds