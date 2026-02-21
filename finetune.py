import os
import math
import torch
import torch.distributed as dist
import hydra
import wandb
import warnings
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch import LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import WandbLogger

from dats import build_dataset
from model import build_model
from loss import build_loss
from tools.callback import build_callback
from tools.validator import build_validator, SingleTaskLMValidator
from tools.utils import trainable_parameter_cnt, update_cfg_for_ddp
from peft import PeftModel, get_peft_model, LoraConfig
from dats.datasets import format_source

# Faster, but less precise
torch.set_float32_matmul_precision("high")
# sets seeds for numpy, torch and python.random.
seed_everything(42, workers=True)

@rank_zero_only
def init_wandb_logger(cfg, save_path):
    logger = WandbLogger(
        name=cfg.wandb.get("name", None),
        project=cfg.wandb.get("project", "default"),
        entity=cfg.wandb.get("entity", None),
        save_dir=save_path,
        tags=cfg.wandb.get("tags", []),
    )
    logger.log_hyperparams(cfg)
    return logger


class Finetuner(LightningModule):
    def __init__(self, cfg, output_path, tokenizer=None):
        super().__init__()

        self.dataset = cfg.dataset.name
        self.batch_size = cfg.get("batch_size")
        self.save_last = cfg.get("save_last", False)
        self.lr = cfg.task.get("learning_rate")
        self.sparsegen = cfg.model.lora.sparsegen_cfg.get("enabled", False)

        if not self.lr:
            if cfg.get("learning_rate") is not None:
                self.lr = cfg.get("learning_rate")
            else:
                raise ValueError(
                    f"Learning rate not found in config file. Please check your config file."
                )
        assert self.lr > 0, f"Learning rate must be positive, got {self.lr}"

        self.weight_decay = cfg.get("weight_decay")
        if cfg.get("scheduler") is not None:
            s = cfg.get("scheduler")
            self.lr_step = s["milestones"]
            self.lr_factor = s["gamma"]

        self.output_path = output_path
    
        self.model = build_model(cfg, cfg.num_labels)
        self.validator = build_validator(cfg)
        self.criterion = build_loss(cfg)

        self.tokenizer = tokenizer
        
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.padding_side = self.tokenizer.padding_side
        # Others
        self.output_attentions = cfg.model.get("output_attentions", False)
        self.log_grad_norm = cfg.get("log_grad_norm", False)

        if cfg.model.get("apply_lora", False):
            lora_config = self.init_lora(cfg)
            if cfg.load_lora_pretrained:
                self.resume_from_checkpoint(cfg, lora_config)
        else:
            if cfg.load_lora_pretrained:
                raise ValueError(
                    f"LoRA is not applied, but you are trying to load LoRA weights. Please check your config file."
                )
        if self.sparsegen:
            self.regular_params = []
            self.sparsegen_params = []
            
            for n, p in self.model.named_parameters():
                if "sparsegen" in n:
                    self.sparsegen_params.append(p)
                else:
                    self.regular_params.append(p)

        self.save_hyperparameters()

        trainable_parameter_cnt(self.model, verbose=False)

    def init_lora(self, cfg):
        lora_meta_config = cfg.model.get("lora", None)
        lora_meta_config = OmegaConf.to_container(lora_meta_config, resolve=True)
        lora_config = LoraConfig(**lora_meta_config)
        self.model = get_peft_model(self.model, lora_config)
        return lora_config

    def configure_optimizers(self):
        if self.sparsegen==True:
            optimizer = torch.optim.AdamW([
                {'params': self.regular_params, 'lr': self.lr},
            ], weight_decay=self.weight_decay)

            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.lr_step, self.lr_factor)
        else:
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.lr_step, self.lr_factor)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        self.clip_gradients(
                            optimizer, 
                            gradient_clip_val=gradient_clip_val,
                            gradient_clip_algorithm=gradient_clip_algorithm
                        )

    def load_peft_weight(self, lora_path=None, lora_config=None, **kwargs):
        if lora_path is not None:
            if isinstance(lora_path, str):
                print(f"Loading lora adapter weights from: {lora_path}")
            elif isinstance(lora_path, dict):
                for k, v in lora_path.items():
                    print(f"Loading lora adapter weights from: {k} -> {v}")
            self.model = PeftModel.from_pretrained(self.model, lora_path, lora_config, **kwargs)
        else:
            raise FileNotFoundError(
                f" Model weight file not found: {lora_path}")
            
    def resume_from_checkpoint(self, cfg, lora_config=None):
        lora_path = cfg.model.get("pretrained_single_lora_paths", None)
        if lora_path:
            lora_path = OmegaConf.to_container(lora_path, resolve=True) 
        else:
            lora_path = cfg.model.get("pretrained_multi_lora_paths", None)
        assert lora_path is not None, \
            f"Pretrained LoRA path not found in config file. Please check your config file."
        model_name = cfg.model.get("model_name", None)

        self.load_peft_weight(
            lora_path, 
            model_name=model_name,
            lora_config=lora_config,
            task=cfg.task.name
        )

    def forward(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        # token_type_ids = batch['token_type_ids']
        if self.criterion:
            outputs = self.model(
                                input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                output_attentions=self.output_attentions
                            )
        else:
            outputs = self.model(
                                input_ids=input_ids, 
                                attention_mask=attention_mask, 
                                labels=labels,
                                output_attentions=self.output_attentions
                            )

        if len(outputs) == 4:
            loss, logits, all_hidden_states, all_attentions = outputs
            all_loraout_dicts = None
        elif len(outputs) == 5:
            loss, logits, all_hidden_states, all_attentions, all_loraout_dicts = outputs
        elif len(outputs) == 6:
            loss, logits, _, all_hidden_states, all_attentions, all_loraout_dicts = outputs
            
        if not loss:
            if self.dataset in ['mmlu_pro', 'arc_e', 'arc_c', 'commonsenseqa', 'openbookqa', 'swag', 'hellaswag', 'flanv2']:
                if self.training and self.criterion:
                    logits = logits[:, :-1]
                    loss = self.criterion(
                                    logits.reshape(-1, logits.size(-1)), 
                                    labels.reshape(-1),
                                    all_loraout_dict=all_loraout_dicts,
                                    attn_mask=attention_mask)
                else:
                    logits = logits[:, -1, :]
            elif self.dataset == 'glue':
                if self.training and self.criterion:
                    loss = self.criterion(
                                    logits.reshape(-1, logits.size(-1)), 
                                    labels.reshape(-1),
                                    all_loraout_dict=all_loraout_dicts,
                                    attn_mask=attention_mask)
                else:
                    logits = logits.reshape(-1, logits.size(-1))

        loss_value = loss.detach() if loss is not None else None
        if self.training and not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            raise Exception("Loss is not finite")

        return loss, logits, all_hidden_states, all_attentions

    def training_step(self, batch, batch_idx):
        loss, logits, all_hidden_states, all_attentions = self.forward(batch, batch_idx)
        log = {}
        log[f"train/loss"] = loss.detach()
        
        if hasattr(self.criterion, 'individual_losses') and self.criterion.individual_losses:
            for name, value in self.criterion.individual_losses.items():
                log[f"train/{name}"] = value.detach()

        if self.sparsegen:
            regular_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log[f"train/lr_r"] = regular_lr
        else:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log[f"train/lr"] = current_lr

        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=self.batch_size)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        results = {}
        if isinstance(self.validator, SingleTaskLMValidator):
            loss, logits, all_hidden_states, all_attentions = self.forward(batch, batch_idx)
            labels = batch['labels']
            res = logits.detach()
            if self.validator:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                preds = self.validator.evaluate(res, self.tokenizer)
                results = self.validator.get_metrics(preds, labels, batch, rank)
                self.validator.update_metrics(results, rank=rank)
        else:
            loss, logits, all_hidden_states, all_attentions = self.forward(batch, batch_idx)
            preds = logits.detach()
            labels = batch['labels']
            if self.validator:
                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                results = self.validator.get_metrics(preds, labels, rank)
                self.validator.update_metrics(results, rank=rank)
        log = {}
        if loss:
            log[f"val/loss"] = loss.detach()

        self.log_dict(
            log,
            logger=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=self.batch_size)

        return {"metrics": results, "loss": loss}
    
    def on_validation_epoch_end(self):
        if self.validator:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            log = {}
            if rank == 0:
                metric = self.validator.get_results()
                log = {f"val/{k}": v for k, v in metric.items()}
                self.log_dict(log, logger=True)
            self.validator.init_metrics()

    def on_after_backward(self):
        if self.log_grad_norm:
            log = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    log[f"grad_norm/{name}"] = grad_norm
            self.log_dict(
                log,
                logger=True,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=True,
                batch_size=self.batch_size)

    @rank_zero_only
    def on_train_end(self):
        if not self.save_last or not self.output_path:
            warnings.warn(
                "Output path is not set. Skipping final model saving."
            )
            return
        save_dir = os.path.join(self.output_path, "peft", "final")
        os.makedirs(save_dir, exist_ok=True)
        
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_dir)
            if hasattr(self, "tokenizer") and self.tokenizer:
                self.tokenizer.save_pretrained(save_dir)
            print(f"Final model saved to: {save_dir}")
        else:
            warnings.warn(
                "Model does not have 'save_pretrained' method. Skipping final model saving."
            )


@hydra.main(config_path="conf", config_name="finetune", version_base="1.3")
def run(cfg: DictConfig):
    save_dir = (f"{cfg.dataset.name}_result/{cfg.task.name}/{cfg.model.name}")
    save_path = os.path.join(cfg.root_dir, save_dir)
    # GPU and Node configuration
    num_nodes = cfg.get("num_nodes", 1)
    if isinstance(cfg.gpu, list):
        num_gpus = len(cfg.gpu)
    elif isinstance(cfg.gpu, int):
        num_gpus = cfg.gpu
    else:
        num_gpus = torch.cuda.device_count()

    cfg = update_cfg_for_ddp(cfg, num_gpus, num_nodes)
    meta_data = build_dataset(cfg)
    train_loader = meta_data['train']
    val_loader = meta_data['val']
    OmegaConf.set_struct(cfg, False)
    cfg.update({
        "num_labels": meta_data['num_labels'],
    })
    OmegaConf.set_struct(cfg, True)
    logger = init_wandb_logger(cfg, save_path)
    print(cfg)
    max_steps = cfg.get("max_steps", -1)
    epochs = cfg.get("num_epochs", None)
    eval_steps = cfg.get("eval_steps", -1)
    cpkt_path = cfg.model.get("cpkt_path", None)
    accumulate_grad_batches = cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg.get("max_grad_norm", None)
    precision = 16 if torch.cuda.is_available() else 32        
    
    if max_steps == -1 and epochs is None:
        raise ValueError("max_steps or epochs must be specified in the config file.")
    if cpkt_path is not None:
        assert cpkt_path and cfg.load_lora_pretrained, \
            "cpkt_path must be specified and load_lora_pretrained must be True to resume training."

    tokenizer=meta_data.get("tokenizer", None)
    saving_cb = build_callback(cfg, tokenizer=tokenizer)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_monitor, saving_cb]
    if num_gpus > 1 and torch.cuda.is_available():
        dist_backend = cfg.get("dist_backend", "nccl")
        strategy = DDPStrategy(process_group_backend=dist_backend)
    else:
        strategy = "auto"

    trainer = Trainer(
        strategy=strategy,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=num_gpus if torch.cuda.is_available() else None,
        num_nodes=num_nodes,
        precision=precision,
        max_steps=max_steps,
        max_epochs=epochs,
        val_check_interval=eval_steps if eval_steps > 0 else None,
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=max_grad_norm,
        deterministic=False,
        num_sanity_val_steps=None,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=False    
        )

    if trainer.is_global_zero and cfg.save:
        if wandb.run is not None and not wandb.run.disabled:
            output_path = logger.experiment.dir
        else:
            output_path = save_path
    else:
        output_path = None

    module = Finetuner(cfg, output_path, tokenizer=tokenizer)

    try:
        trainer.fit(module, train_loader, val_loader, ckpt_path=cpkt_path)
        # trainer.validate(module, val_loader)
    except Exception as e:
        print(f"Exception caught: {e}")
        raise e

if __name__ == "__main__":
    run()