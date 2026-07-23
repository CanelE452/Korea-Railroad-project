import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from smoke.utils.metric_logger import MetricLogger
from smoke.utils.comm import get_world_size
from smoke.layers.utils import select_point_of_interest


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.update(model, decay=0.0)

    @staticmethod
    def _state_dict(model):
        return model.module.state_dict() if hasattr(model, "module") else model.state_dict()

    @torch.no_grad()
    def update(self, model, decay=None):
        if decay is None:
            decay = self.decay
        state_dict = self._state_dict(model)
        for key, value in state_dict.items():
            if not torch.is_floating_point(value):
                continue
            value = value.detach()
            if key not in self.shadow:
                self.shadow[key] = value.clone()
            else:
                self.shadow[key].mul_(decay).add_(value, alpha=1.0 - decay)

    def state_dict(self):
        return {key: value.cpu().clone() for key, value in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {
            key: value.detach().clone()
            for key, value in state_dict.items()
            if torch.is_floating_point(value)
        }


def save_ema_checkpoint(checkpointer, name, model_ema, arguments):
    if not checkpointer.save_dir or not checkpointer.save_to_disk:
        return
    save_file = os.path.join(checkpointer.save_dir, "{}_ema.pth".format(name))
    data = {"model": model_ema.state_dict()}
    for key, value in arguments.items():
        if key != "model_ema":
            data[key] = value
    checkpointer.logger.info("Saving EMA checkpoint to {}".format(save_file))
    torch.save(data, save_file)


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _raw_predictions(model, images):
    model = _unwrap_model(model)
    features = model.backbone(images.tensors)
    return model.heads.predictor(features)


def _build_teacher(cfg, device):
    if not cfg.DISTILL.ENABLED:
        return None
    if not cfg.DISTILL.TEACHER_CONFIG:
        raise ValueError("DISTILL.TEACHER_CONFIG must be set when distillation is enabled")
    if not cfg.DISTILL.TEACHER_WEIGHT:
        raise ValueError("DISTILL.TEACHER_WEIGHT must be set when distillation is enabled")

    from smoke.config import cfg as base_cfg
    from smoke.modeling.detector import build_detection_model
    from smoke.utils.check_point import DetectronCheckpointer

    teacher_config = cfg.DISTILL.TEACHER_CONFIG
    if not os.path.isabs(teacher_config):
        teacher_config = os.path.abspath(teacher_config)

    teacher_weight = cfg.DISTILL.TEACHER_WEIGHT
    if not os.path.isabs(teacher_weight):
        teacher_weight = os.path.abspath(teacher_weight)

    teacher_cfg = base_cfg.clone()
    teacher_cfg.merge_from_file(teacher_config)
    teacher_cfg.defrost()
    teacher_cfg.MODEL.DEVICE = cfg.MODEL.DEVICE
    teacher_cfg.DATASETS.DETECT_CLASSES = cfg.DATASETS.DETECT_CLASSES
    teacher_cfg.freeze()

    teacher = build_detection_model(teacher_cfg)
    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    checkpointer = DetectronCheckpointer(teacher_cfg, teacher, save_dir="")
    checkpointer.load(teacher_weight, use_latest=False)
    return teacher


def _compute_distillation_losses(cfg, student_predictions, teacher_predictions, targets):
    student_heatmap, student_regression = student_predictions
    teacher_heatmap, teacher_regression = teacher_predictions
    teacher_heatmap = teacher_heatmap.detach()
    teacher_regression = teacher_regression.detach()

    distill_hm = F.mse_loss(student_heatmap, teacher_heatmap, reduction="mean")

    batch, channel = student_regression.shape[0], student_regression.shape[1]
    proj_points = torch.stack([target.get_field("proj_p") for target in targets]).to(
        device=student_regression.device
    )
    reg_mask = torch.stack([target.get_field("reg_mask") for target in targets]).to(
        device=student_regression.device
    ).bool()

    student_pois = select_point_of_interest(batch, proj_points, student_regression).view(batch, -1, channel)
    teacher_pois = select_point_of_interest(batch, proj_points, teacher_regression).view(batch, -1, channel)

    if reg_mask.sum() == 0:
        distill_reg = student_regression.sum() * 0.0
    else:
        distill_reg = F.smooth_l1_loss(
            student_pois[reg_mask],
            teacher_pois[reg_mask],
            beta=cfg.MODEL.SMOKE_HEAD.SMOOTH_L1_BETA,
            reduction="mean",
        )

    return {
        "distill_hm_loss": distill_hm * cfg.DISTILL.HM_WEIGHT,
        "distill_reg_loss": distill_reg * cfg.DISTILL.REG_WEIGHT,
    }


def do_train(
        cfg,
        distributed,
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
):
    logger = logging.getLogger("smoke.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter=" ")
    max_iter = cfg.SOLVER.MAX_ITERATION
    start_iter = arguments["iteration"]
    model_ema = None
    if cfg.SOLVER.USE_EMA:
        model_ema = ModelEMA(model, cfg.SOLVER.EMA_DECAY)
        if "model_ema" in arguments:
            model_ema.load_state_dict(arguments["model_ema"])
        logger.info("EMA enabled with decay {}".format(cfg.SOLVER.EMA_DECAY))
    teacher_model = _build_teacher(cfg, device)
    if teacher_model is not None:
        logger.info(
            "Distillation enabled: teacher={}, hm_weight={}, reg_weight={}, gt_weight={}".format(
                cfg.DISTILL.TEACHER_WEIGHT,
                cfg.DISTILL.HM_WEIGHT,
                cfg.DISTILL.REG_WEIGHT,
                cfg.DISTILL.GT_WEIGHT,
            )
        )
    model.train()
    start_training_time = time.time()
    end = time.time()

    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        data_time = time.time() - end
        iteration += 1
        arguments["iteration"] = iteration

        images = data["images"].to(device)
        targets = [target.to(device) for target in data["targets"]]

        if teacher_model is not None:
            loss_dict, student_predictions = model(images, targets, return_predictions=True)
            loss_dict = {
                key: value * cfg.DISTILL.GT_WEIGHT
                for key, value in loss_dict.items()
            }
            with torch.no_grad():
                teacher_predictions = _raw_predictions(teacher_model, images)
            loss_dict.update(
                _compute_distillation_losses(
                    cfg,
                    student_predictions,
                    teacher_predictions,
                    targets,
                )
            )
        else:
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        if model_ema is not None:
            model_ema.update(model)
        scheduler.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 10 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.8f}",
                        "max men: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
                )
            )
        should_save_periodic = (
            checkpoint_period > 0
            and iteration % checkpoint_period == 0
            and iteration != max_iter
        )
        should_save_step = iteration in cfg.SOLVER.STEPS and iteration != max_iter

        if should_save_periodic or should_save_step:
            if model_ema is not None:
                arguments["model_ema"] = model_ema.state_dict()
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if model_ema is not None:
                save_ema_checkpoint(checkpointer, "model_{:07d}".format(iteration), model_ema, arguments)
        if iteration == max_iter:
            if model_ema is not None:
                arguments["model_ema"] = model_ema.state_dict()
            checkpointer.save("model_final", **arguments)
            if model_ema is not None:
                save_ema_checkpoint(checkpointer, "model_final", model_ema, arguments)
        # todo: add evaluations here
        # if iteration % evaluate_period == 0:
        # test_net.main()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
