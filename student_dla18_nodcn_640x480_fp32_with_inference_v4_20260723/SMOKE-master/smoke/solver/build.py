import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        params += [{"params": [value], "lr": lr, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}]

    optimizer_name = cfg.SOLVER.OPTIMIZER.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=cfg.SOLVER.BASE_LR)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR)
    else:
        raise ValueError("Unsupported optimizer: {}".format(cfg.SOLVER.OPTIMIZER))

    return optimizer


def make_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        cfg.SOLVER.STEPS
    )
