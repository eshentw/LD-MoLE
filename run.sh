python finetune.py \
    --config-name=finetune \
    model=llama3.2_3b \ # base model: llama3.1_8b, llama3.2_1b, llama3.2_3b, qwen3_0.6b, qwen3_1.7b
    model.lora.modules_to_save='["score", "sparsegen"]' \
    model.lora.sparsegen_cfg.enabled=true \ # whether to use sparsegen mlp
    model.lora.sparsegen_cfg.hidden_sizes=512 \ # intermidiate hidden size of sparsegen mlp
    gpu=1 \
    batch_size=8 \
    val_batch_size=16 \
    dataset=arc_c \ # For mrpc and rte use "glue" for dataset and same as task otherwise, e.g. "arc_c", "mmlu_pro", "hellaswag"
    task=arc_c \
    ce_loss_coef=1.0 \ # cross entropy loss coefficient
    lb_loss_coef=1.0 \ # load balancing loss coefficient
    reg_loss_coef=0.0 \ # regularization loss coefficient
    gradient_accumulation_steps=1 \
    num_epochs=3 \
    dist_backend=gloo