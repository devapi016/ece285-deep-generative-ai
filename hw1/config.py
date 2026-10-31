"""Shared paths and hyperparameters for the chest X-ray VAE (single source of truth)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "chest_xray"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_PATH = RESULTS_DIR / "vae.pt"
HISTORY_PATH = RESULTS_DIR / "loss_history.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# The Kaggle "val" split holds only 16 images, so it is folded into training and
# the 624-image test split serves as the held-out set for validation loss and FID.
TRAIN_SPLITS = ("train", "val")
HELDOUT_SPLIT = "test"
CLASS_NAMES = ("NORMAL", "PNEUMONIA")
# One class suffices for the assignment; PNEUMONIA has ~3x more images than NORMAL.
TRAINING_CLASS = "PNEUMONIA"

# Data
IMAGE_SIZE = 128
NUM_DECODE_WORKERS = 16

# Model
LATENT_DIM = 128
ENCODER_CHANNELS = (32, 64, 128, 256, 512)  # each stage halves the spatial size

# Optimisation
SEED = 0
BATCH_SIZE = 64
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
LR_DECAY_FACTOR = 0.01  # cosine schedule ends at LEARNING_RATE * LR_DECAY_FACTOR
KL_WEIGHT = 1.0
KL_WARMUP_EPOCHS = 20  # linear KL annealing avoids early posterior collapse

# Evaluation. Ignite's FID needs equally sized real/candidate sets, so every candidate
# set is matched to the held-out images and redrawn NUM_METRIC_REPEATS times.
NUM_METRIC_REPEATS = 5
INCEPTION_INPUT_SIZE = 299
METRIC_BATCH_SIZE = 100
NUM_GRID_IMAGES = 48
