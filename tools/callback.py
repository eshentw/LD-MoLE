import os
import torch
import torch.distributed as dist
import warnings
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only


def build_callback(cfg, tokenizer=None):
    monitor = cfg.task.get('monitor', 'val/loss')
    if cfg.save:
        if cfg.dataset.name == "glue":
            mode = 'max' if 'accuracy' in monitor else 'min'
            saving_cb = DefaultSaveBestCallback(
                    monitor=monitor,
                    mode=mode)
        else:
            saving_cb = ResumeSaveBestCallback(
                    monitor=monitor,
                    mode='max',
                    tokenizer=tokenizer)
    else:
        saving_cb = NoSaveCallback()
    return saving_cb


class NoSaveCallback(Callback):
    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        print("No save path provided, skipping checkpoint saving.")
        return {}


class DefaultSaveBestCallback(Callback):
    def __init__(self, monitor, mode="min"):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            metrics = trainer.callback_metrics
            current_score = metrics.get(self.monitor)
            if current_score is None:
                warnings.warn(
                    f"Monitor metric '{self.monitor}' not found in callback metrics. Skipping best model saving.")
            else:
                current_score = current_score.item() if torch.is_tensor(current_score) else current_score

                if self.best_score is None or \
                (self.mode == "min" and current_score < self.best_score) or \
                (self.mode == "max" and current_score > self.best_score):
                    self.best_score = current_score
                    best_dir = os.path.join(pl_module.output_path, "peft", "best")
                    os.makedirs(best_dir, exist_ok=True)
                    pl_module.model.save_pretrained(best_dir)
                    if hasattr(pl_module, "tokenizer") and pl_module.tokenizer:
                        pl_module.tokenizer.save_pretrained(best_dir)
                    print(f"Best model saved to: {best_dir}")

        if dist.is_initialized() and dist.is_available():
            dist.barrier()


class ResumeSaveBestCallback(Callback):
    '''
    Callback to save the best model and the resume checkpoint
    based on a monitored metric during training.
    '''
    def __init__(self, monitor, mode, tokenizer=None):
        super().__init__()
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.tokenizer = tokenizer

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.best_score = checkpoint.get("best_score", None)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["best_score"] = self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            step_save_dir = os.path.join(
                pl_module.output_path, "peft", f"last_checkpoints")
            os.makedirs(step_save_dir, exist_ok=True)
            step_ckpt_path = os.path.join(step_save_dir, "checkpoint.ckpt")
        else:
            step_ckpt_path = None
        # this only save ckpt on global zero
        trainer.save_checkpoint(step_ckpt_path)

        if trainer.is_global_zero:
            if hasattr(pl_module.model, "save_pretrained"):
                pl_module.model.save_pretrained(step_save_dir)
                if hasattr(pl_module, "tokenizer") and pl_module.tokenizer:
                    pl_module.tokenizer.save_pretrained(step_save_dir)

            metrics = trainer.callback_metrics
            current_score = metrics.get(self.monitor)
            if current_score is None:
                warnings.warn(
                    f"Monitor metric '{self.monitor}' not found in callback metrics. Skipping best model saving.")
            else:
                current_score = current_score.item() if torch.is_tensor(current_score) else current_score

                if self.best_score is None or \
                (self.mode == "min" and current_score < self.best_score) or \
                (self.mode == "max" and current_score > self.best_score):
                    self.best_score = current_score
                    best_dir = os.path.join(pl_module.output_path, "peft", "best")
                    os.makedirs(best_dir, exist_ok=True)
                    pl_module.model.save_pretrained(best_dir)
                    if hasattr(pl_module, "tokenizer") and pl_module.tokenizer:
                        pl_module.tokenizer.save_pretrained(best_dir)
                    print(f"Best model saved to: {best_dir}")

        if dist.is_initialized() and dist.is_available():
            dist.barrier()   
            

class ResumeSaveAllCallback(Callback):
    '''
    Callback to save all the model and resume checkpoint
    based on a monitored metric during training.
    '''
    def __init__(self, monitor, mode, tokenizer=None, on_epoch=False):
        super().__init__()
        self.on_epoch = on_epoch
        self.monitor = monitor
        self.mode = mode
        self.best_score = None
        self.tokenizer = tokenizer

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        self.best_score = checkpoint.get("best_score", None)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["best_score"] = self.best_score

    def on_validation_end(self, trainer, pl_module):
        if trainer.is_global_zero:
            if self.on_epoch:
                step_save_dir = os.path.join(
                    pl_module.output_path, "peft", f"epoch_{trainer.current_epoch}")
            else:
                step_save_dir = os.path.join(
                    pl_module.output_path, "peft", f"step_{trainer.global_step}")
            os.makedirs(step_save_dir, exist_ok=True)
            step_ckpt_path = os.path.join(step_save_dir, "checkpoint.ckpt")
        else:
            step_ckpt_path = None
        # this only save ckpt on global zero
        trainer.save_checkpoint(step_ckpt_path)

        if trainer.is_global_zero:
            if hasattr(pl_module.model, "save_pretrained"):
                pl_module.model.save_pretrained(step_save_dir)
                if hasattr(pl_module, "tokenizer") and pl_module.tokenizer:
                    pl_module.tokenizer.save_pretrained(step_save_dir)

            metrics = trainer.callback_metrics
            current_score = metrics.get(self.monitor)
            if current_score is None:
                warnings.warn(
                    f"Monitor metric '{self.monitor}' not found in callback metrics. Skipping best model saving.")
            else:
                current_score = current_score.item() if torch.is_tensor(current_score) else current_score

                if self.best_score is None or \
                (self.mode == "min" and current_score < self.best_score) or \
                (self.mode == "max" and current_score > self.best_score):
                    self.best_score = current_score
                    best_dir = os.path.join(pl_module.output_path, "peft", "best")
                    os.makedirs(best_dir, exist_ok=True)
                    pl_module.model.save_pretrained(best_dir)
                    if hasattr(pl_module, "tokenizer") and pl_module.tokenizer:
                        pl_module.tokenizer.save_pretrained(best_dir)
                    print(f"Best model saved to: {best_dir}")

        if dist.is_initialized() and dist.is_available():
            dist.barrier()   
