"""Trains the VAE on chest X-rays and records per-epoch loss curves.

Usage: python train.py
"""

import json
import time

import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH,
    HISTORY_PATH,
    KL_WARMUP_EPOCHS,
    KL_WEIGHT,
    LEARNING_RATE,
    LR_DECAY_FACTOR,
    NUM_EPOCHS,
    RESULTS_DIR,
    SEED,
)
from data import make_loaders
from model import VAE, elbo_terms

LOSS_KEYS = ("reconstruction", "kl", "negative_elbo")


def kl_weight_for_epoch(epoch: int) -> float:
    """Linearly anneals the KL weight from ~0 to KL_WEIGHT over the warm-up epochs."""
    return KL_WEIGHT * min(1.0, (epoch + 1) / KL_WARMUP_EPOCHS)


def _empty_totals() -> dict[str, float]:
    return dict.fromkeys(LOSS_KEYS, 0.0)


def _accumulate(totals: dict[str, float], reconstruction: torch.Tensor, kl: torch.Tensor) -> None:
    totals["reconstruction"] += reconstruction.item()
    totals["kl"] += kl.item()
    totals["negative_elbo"] += (reconstruction + kl).item()


def _average(totals: dict[str, float], num_batches: int) -> dict[str, float]:
    return {key: value / num_batches for key, value in totals.items()}


def train_one_epoch(
    model: VAE, loader: DataLoader, optimizer: torch.optim.Optimizer, kl_weight: float, device: torch.device
) -> dict[str, float]:
    model.train()
    totals = _empty_totals()
    for (images,) in loader:
        images = images.to(device, non_blocking=True)
        logits, mean, log_variance = model(images)
        reconstruction, kl = elbo_terms(logits, images, mean, log_variance)
        loss = reconstruction + kl_weight * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        _accumulate(totals, reconstruction, kl)
    return _average(totals, len(loader))


@torch.no_grad()
def evaluate_loss(model: VAE, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Held-out negative ELBO with the full KL weight (the quantity a VAE bounds)."""
    model.eval()
    totals = _empty_totals()
    for (images,) in loader:
        images = images.to(device, non_blocking=True)
        logits, mean, log_variance = model(images)
        _accumulate(totals, *elbo_terms(logits, images, mean, log_variance))
    return _average(totals, len(loader))


def main() -> None:
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, heldout_loader = make_loaders()
    model = VAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LEARNING_RATE * LR_DECAY_FACTOR
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training on {len(train_loader.dataset)} images, {num_parameters / 1e6:.2f}M parameters, {device}")

    history = {"train": [], "heldout": [], "kl_weight": []}
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        kl_weight = kl_weight_for_epoch(epoch)
        train_losses = train_one_epoch(model, train_loader, optimizer, kl_weight, device)
        heldout_losses = evaluate_loss(model, heldout_loader, device)
        scheduler.step()

        history["train"].append(train_losses)
        history["heldout"].append(heldout_losses)
        history["kl_weight"].append(kl_weight)
        print(
            f"epoch {epoch + 1:3d}/{NUM_EPOCHS}  beta={kl_weight:.2f}  "
            f"train -ELBO {train_losses['negative_elbo']:.1f} (KL {train_losses['kl']:.1f})  "
            f"held-out -ELBO {heldout_losses['negative_elbo']:.1f}  [{time.time() - start_time:.0f}s]"
        )

    history["training_seconds"] = time.time() - start_time
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    HISTORY_PATH.write_text(json.dumps(history, indent=2))
    print(f"Saved {CHECKPOINT_PATH} and {HISTORY_PATH}")


if __name__ == "__main__":
    main()
