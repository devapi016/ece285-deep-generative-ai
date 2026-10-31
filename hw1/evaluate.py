"""Generates chest X-rays with the trained VAE and scores them with FID and Inception Score.

Metrics use PyTorch-Ignite's FID / InceptionScore as in
https://pytorch-ignite.ai/blog/gan-evaluation-with-fid-and-is/ : grayscale images in
[0, 1] are repeated to 3 channels and bilinearly resized to 299x299 for InceptionV3.

Usage: python evaluate.py   (after python train.py)
"""

import json
from collections.abc import Callable

import torch
import torch.nn.functional as F
from ignite.metrics import FID, InceptionScore

from config import (
    CHECKPOINT_PATH,
    HELDOUT_SPLIT,
    HISTORY_PATH,
    INCEPTION_INPUT_SIZE,
    METRIC_BATCH_SIZE,
    METRICS_PATH,
    NUM_GRID_IMAGES,
    NUM_METRIC_REPEATS,
    RESULTS_DIR,
    SEED,
    TRAIN_SPLITS,
)
from data import load_images
from model import VAE
from visualize import plot_loss_curves, save_grid

GRID_IMAGES_PER_ROW = 16
NUM_RECONSTRUCTION_PAIRS = 16
RGB_CHANNELS = 3

# Draws `count` candidate images in [0, 1] using the given random generator.
CandidateSource = Callable[[int, torch.Generator], torch.Tensor]


def to_inception_input(images: torch.Tensor) -> torch.Tensor:
    resized = F.interpolate(
        images, size=(INCEPTION_INPUT_SIZE, INCEPTION_INPUT_SIZE), mode="bilinear", align_corners=False
    )
    return resized.repeat(1, RGB_CHANNELS, 1, 1)


def score_against_reference(
    candidates: torch.Tensor, reference: torch.Tensor, device: torch.device
) -> tuple[float, float]:
    """Returns (FID of candidates vs. reference, Inception Score of candidates)."""
    fid = FID(device=device)
    inception_score = InceptionScore(device=device)
    for candidate_batch, reference_batch in zip(
        candidates.split(METRIC_BATCH_SIZE), reference.split(METRIC_BATCH_SIZE)
    ):
        candidate_batch = to_inception_input(candidate_batch.to(device))
        fid.update((candidate_batch, to_inception_input(reference_batch.to(device))))
        inception_score.update(candidate_batch)
    return float(fid.compute()), float(inception_score.compute())


def inception_score_only(images: torch.Tensor, device: torch.device) -> float:
    inception_score = InceptionScore(device=device)
    for batch in images.split(METRIC_BATCH_SIZE):
        inception_score.update(to_inception_input(batch.to(device)))
    return float(inception_score.compute())


@torch.no_grad()
def reconstruct(model: VAE, images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Deterministic reconstruction through the posterior mean (no sampling noise)."""
    reconstructions = []
    for batch in images.split(METRIC_BATCH_SIZE):
        mean, _ = model.encode(batch.to(device))
        reconstructions.append(torch.sigmoid(model.decode_logits(mean)).cpu())
    return torch.cat(reconstructions)


def build_candidate_sources(model: VAE, train_images: torch.Tensor, device: torch.device) -> dict[str, CandidateSource]:
    def random_train_subset(count: int, generator: torch.Generator) -> torch.Tensor:
        return train_images[torch.randperm(len(train_images), generator=generator)[:count]]

    def vae_samples(count: int, generator: torch.Generator) -> torch.Tensor:
        latents = torch.randn(count, model.latent_dim, generator=generator).to(device)
        with torch.no_grad():
            return torch.sigmoid(model.decode_logits(latents)).cpu()

    return {
        "real_train_images": random_train_subset,
        "vae_reconstructions": lambda count, generator: reconstruct(model, random_train_subset(count, generator), device),
        "vae_samples": vae_samples,
    }


def compute_metrics(
    sources: dict[str, CandidateSource], heldout_images: torch.Tensor, device: torch.device
) -> dict[str, dict[str, float]]:
    num_reference = len(heldout_images)
    metrics = {}
    for name, draw_candidates in sources.items():
        fids, inception_scores = [], []
        for repeat in range(NUM_METRIC_REPEATS):
            generator = torch.Generator().manual_seed(SEED + repeat)
            fid, inception_score = score_against_reference(draw_candidates(num_reference, generator), heldout_images, device)
            fids.append(fid)
            inception_scores.append(inception_score)
        fids, inception_scores = torch.tensor(fids), torch.tensor(inception_scores)
        metrics[name] = {
            "fid_mean": fids.mean().item(),
            "fid_std": fids.std().item(),
            "is_mean": inception_scores.mean().item(),
            "is_std": inception_scores.std().item(),
        }
        print(f"{name:22s} FID {metrics[name]['fid_mean']:7.2f} ± {metrics[name]['fid_std']:.2f}   "
              f"IS {metrics[name]['is_mean']:.3f} ± {metrics[name]['is_std']:.3f}")

    metrics["real_heldout_images"] = {"is_mean": inception_score_only(heldout_images, device)}
    metrics["num_images_per_set"] = num_reference
    return metrics


def save_figures(model: VAE, heldout_images: torch.Tensor, device: torch.device) -> None:
    plot_loss_curves(json.loads(HISTORY_PATH.read_text()), RESULTS_DIR / "loss_curves.png")

    torch.manual_seed(SEED)
    save_grid(model.sample(NUM_GRID_IMAGES, device).cpu(), RESULTS_DIR / "generated_samples.png", GRID_IMAGES_PER_ROW)

    # Top row: held-out originals; bottom row: their reconstructions.
    originals = heldout_images[torch.randperm(len(heldout_images))[:NUM_RECONSTRUCTION_PAIRS]]
    pairs = torch.cat([originals, reconstruct(model, originals, device)])
    save_grid(pairs, RESULTS_DIR / "reconstructions.png", NUM_RECONSTRUCTION_PAIRS)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    train_images = load_images(TRAIN_SPLITS)
    heldout_images = load_images((HELDOUT_SPLIT,))

    save_figures(model, heldout_images, device)
    metrics = compute_metrics(build_candidate_sources(model, train_images, device), heldout_images, device)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved figures and {METRICS_PATH}")


if __name__ == "__main__":
    main()
