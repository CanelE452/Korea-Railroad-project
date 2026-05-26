"""self_train.py — 학습 step 모듈.

train_one_epoch: synthetic GT + pseudo-labeled real 혼합 한 epoch 학습.
  - synthetic: MSE(belief) + MSE(affinity) (all 6 stage 합산)
  - pseudo:    MSE(belief) only (affinity GT 없음), λ_real scale
"""
from __future__ import annotations
import numpy as np
import torch
import torch.utils.data as data


def train_one_epoch(model, optimizer, syn_loader, pseudo_dataset,
                    config, device, epoch, round_idx):
    """synthetic + pseudo-real 혼합 한 epoch 학습. dict 통계 반환."""
    lambda_real = config["self_training"]["lambda_real"]
    log_interval = config["logging"]["log_interval"]
    model.train()

    pseudo_loader = None
    pseudo_iter = None
    if len(pseudo_dataset) > 0:
        pseudo_loader = data.DataLoader(
            pseudo_dataset,
            batch_size=min(config["self_training"]["batch_size"], len(pseudo_dataset)),
            shuffle=True, num_workers=0, drop_last=False,
        )
        pseudo_iter = iter(pseudo_loader)

    losses_syn = []
    losses_real = []
    losses_total = []

    for batch_idx, syn_batch in enumerate(syn_loader):
        optimizer.zero_grad()

        # --- Synthetic loss (MSE belief + affinity, 6 stage) ---
        syn_img = syn_batch["img"].to(device)
        syn_beliefs = syn_batch["beliefs"].to(device)
        syn_affinities = syn_batch["affinities"].to(device)

        out_bel, out_aff = model(syn_img)
        loss_syn = torch.tensor(0.0, device=device)
        for stage in range(len(out_bel)):
            loss_syn += ((out_bel[stage] - syn_beliefs) ** 2).mean()
            loss_syn += ((out_aff[stage] - syn_affinities) ** 2).mean()
        losses_syn.append(loss_syn.item())

        # --- Pseudo-labeled real loss (belief only, λ scale) ---
        loss_real = torch.tensor(0.0, device=device)
        if pseudo_loader is not None and len(pseudo_dataset) > 0:
            try:
                real_batch = next(pseudo_iter)
            except StopIteration:
                pseudo_iter = iter(pseudo_loader)
                real_batch = next(pseudo_iter)
            real_img = real_batch["img"].to(device)
            real_beliefs = real_batch["beliefs"].to(device)
            out_bel_real, _ = model(real_img)
            for stage in range(len(out_bel_real)):
                loss_real += ((out_bel_real[stage] - real_beliefs) ** 2).mean()
            loss_real = lambda_real * loss_real
        losses_real.append(loss_real.item())

        total_loss = loss_syn + loss_real
        total_loss.backward()
        optimizer.step()
        losses_total.append(total_loss.item())

        if (batch_idx + 1) % log_interval == 0:
            print(f"  Round {round_idx} Epoch {epoch} "
                  f"[{batch_idx + 1}/{len(syn_loader)}] "
                  f"loss_syn={loss_syn.item():.6f} "
                  f"loss_real={loss_real.item():.6f} "
                  f"total={total_loss.item():.6f}")

    return {
        "loss_syn": float(np.mean(losses_syn)),
        "loss_real": float(np.mean(losses_real)),
        "loss_total": float(np.mean(losses_total)),
    }
