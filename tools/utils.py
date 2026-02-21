import torch
from collections.abc import Mapping


def parameter_cnt(model):
    return sum(p.numel() for p in model.parameters())


def trainable_parameter_cnt(model, verbose=False):
    param_cnt = 0
    total_params = parameter_cnt(model)
    for n, p in model.named_parameters():
        if p.requires_grad:
            if verbose:
                print(f"Trainable parameter: {n}, shape: {p.shape}")
            param_cnt += p.numel()
    print(f"Total trainable parameters: {param_cnt}, Percentage: {param_cnt / total_params * 100:.2f}%")


def nested_numpify(tensors):
    "Numpify `tensors` (even if it's a nested list/tuple/dict of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(nested_numpify(t) for t in tensors)
    if isinstance(tensors, Mapping):
        return type(tensors)({k: nested_numpify(t) for k, t in tensors.items()})

    t = tensors.cpu()
    if t.dtype == torch.bfloat16:
        # As of Numpy 1.21.4, NumPy does not support bfloat16 (see
        # https://github.com/numpy/numpy/blob/a47ecdea856986cd60eabbd53265c2ca5916ad5d/doc/source/user/basics.types.rst ).
        # Until Numpy adds bfloat16, we must convert float32.
        t = t.to(torch.float32)
    return t.numpy()


def update_cfg_for_ddp(cfg, num_gpus, num_nodes):
    world_size = num_gpus * num_nodes
    assert cfg.batch_size % world_size == 0, \
        f"Batch size {cfg.batch_size} must be divisible by world size {world_size}."
    cfg.update({
        "batch_size": cfg.batch_size // world_size if num_gpus > 1 else cfg.batch_size,
    })
    if cfg.get("val_batch_size") is not None:
        assert cfg.val_batch_size % world_size == 0, \
            f"Validation batch size {cfg.val_batch_size} must be divisible by world size {world_size}."
        cfg.update({
            "val_batch_size": cfg.val_batch_size // world_size if num_gpus > 1 else cfg.val_batch_size,
        })
    return cfg