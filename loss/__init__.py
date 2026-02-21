from .loss import *

def build_loss(cfg):
    criterion = None
    dataset = cfg.dataset.name
    if dataset in ['mmlu_pro', 'arc_e', 'arc_c', 'commonsenseqa', 'openbookqa', 'swag', 'hellaswag'] or \
        dataset == 'glue' or dataset == 'flanv2':
        criterion = CompositeLoss()
        ce_loss_fn = CrossEntropyLoss(
            ignore_index=cfg.dataset.get('ignore_index', -100),
            label_smoothing=cfg.task.get('label_smoothing', 0.1),
        )
        lb_loss_fn = LoadBalancingLoss()
        rz_loss_fn = Router_z_loss()
        ce_loss_coef = cfg.get('ce_loss_coef', 1.0)
        lb_loss_coef = cfg.get('lb_loss_coef', 0.0)
        rz_loss_coef = cfg.get('rz_loss_coef', 0.0)
        reg_loss_coef = cfg.get('reg_loss_coef', 0.0)
        lam_loss_coef = cfg.get('lam_loss_coef', 0.0)
        if reg_loss_coef > 0:
            reg_loss_fn = RegularizationLoss()
            criterion.add_loss(reg_loss_fn, weight=reg_loss_coef)
        criterion.add_loss(ce_loss_fn, weight=ce_loss_coef)
        if lb_loss_coef > 0:
            criterion.add_loss(lb_loss_fn, weight=lb_loss_coef)
        if rz_loss_coef > 0:
            criterion.add_loss(rz_loss_fn, weight=rz_loss_coef)
        if lam_loss_coef > 0:
            lam_loss_fn = LamSparseLoss()
            criterion.add_loss(lam_loss_fn, weight=lam_loss_coef)

    return criterion