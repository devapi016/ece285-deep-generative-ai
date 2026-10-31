"""Figure helpers for the report: loss curves and image grids."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image

FIGURE_DPI = 200
GRID_PADDING = 2
TRAIN_COLOR = "#1f6fb4"
HELDOUT_COLOR = "#d9711c"
KL_WEIGHT_COLOR = "#6b6b6b"
Y_MARGIN_FRACTION = 0.05


def plot_loss_curves(history: dict, output_path: Path) -> None:
    """Plots negative ELBO, reconstruction and KL terms for train vs. held-out data."""
    epochs = range(1, len(history["train"]) + 1)
    panels = (
        ("negative_elbo", "Negative ELBO"),
        ("reconstruction", "Reconstruction (BCE)"),
        ("kl", "KL divergence"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(10, 2.6))
    for axis, (key, title) in zip(axes, panels):
        axis.plot(epochs, [entry[key] for entry in history["train"]], color=TRAIN_COLOR, label="train")
        axis.plot(epochs, [entry[key] for entry in history["heldout"]], color=HELDOUT_COLOR, label="held-out (test)")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)

        # The first epoch is far above the rest; excluding it from the y-range keeps
        # the late-epoch train/held-out gap visible.
        later_values = [entry[key] for split in ("train", "heldout") for entry in history[split][1:]]
        axis.set_ylim(min(later_values) - Y_MARGIN_FRACTION * (max(later_values) - min(later_values)), max(later_values))

    axes[0].set_ylabel("nats per image")
    kl_weight_axis = axes[-1].twinx()
    kl_weight_axis.plot(epochs, history["kl_weight"], color=KL_WEIGHT_COLOR, linestyle="--", label=r"KL weight $\beta$")
    kl_weight_axis.set_ylim(0, 1.05)
    kl_weight_axis.tick_params(labelsize=8)

    handles = axes[0].get_legend_handles_labels()[0] + kl_weight_axis.get_legend_handles_labels()[0]
    labels = axes[0].get_legend_handles_labels()[1] + kl_weight_axis.get_legend_handles_labels()[1]
    figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.12))
    figure.tight_layout()
    figure.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_grid(images: torch.Tensor, output_path: Path, images_per_row: int) -> None:
    """Saves images in [0, 1] as a single tiled PNG."""
    save_image(images, output_path, nrow=images_per_row, padding=GRID_PADDING, pad_value=1.0)
