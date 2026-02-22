# Use physical GPUs 4,5,6,7. Inside this process they are remapped to cuda:0,1,2,3.
# Disable Weights & Biases logging for this run.
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=4,5,6,7 python finetune.py \
    --config-name=finetune \
    model=qwen3_0.6b \
    model.lora.modules_to_save='["score", "sparsegen"]' \
    model.lora.sparsegen_cfg.enabled=true \
    model.lora.sparsegen_cfg.hidden_sizes=512 \
    gpu=4 \
    batch_size=32 \
    val_batch_size=32 \
    dataset=arc_c \
    task=arc_c \
    ce_loss_coef=1.0 \
    lb_loss_coef=1.0 \
    reg_loss_coef=0.0 \
    gradient_accumulation_steps=1 \
    num_epochs=3 \
    dist_backend=nccl
