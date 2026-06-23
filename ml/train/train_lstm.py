"""
Train LSTM crop prediction model for a given region.

Reads crop yield + weather history from TimescaleDB.  Builds per-hex
multi-season sequences (seq_len=8 seasons ≈ 4 years) and trains an LSTM
to predict per-crop yield at the next season.

Prerequisites:
    - scripts/ingest_apy.py must have been run (crop_yield_obs populated)
    - scripts/ingest_era5.py must have been run (weather_obs populated)
    - ml/train/train_sarimax.py must have been run first
      (scaler.pkl + crop_index.json + model_meta.json are reused)

Saves:
    ml/artifacts/{region_code}/
        lstm_model.pt       — PyTorch state_dict
        lstm_config.json    — architecture hyperparameters

Usage:
    python ml/train/train_lstm.py --region IN
    python ml/train/train_lstm.py --region IN --epochs 50 --hidden 256 --seq-len 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from ml.pipeline import FEATURE_COLUMNS, build_training_bundle

logger = structlog.get_logger(__name__)

_ARTIFACTS_ROOT = Path("ml/artifacts")

# Default hyperparameters
_DEFAULT_HIDDEN   = 128
_DEFAULT_LAYERS   = 2
_DEFAULT_SEQ_LEN  = 8      # 8 seasons ≈ 4 kharif+rabi cycle years
_DEFAULT_EPOCHS   = 40
_DEFAULT_LR       = 1e-3
_DEFAULT_BATCH    = 64
_DEFAULT_DROPOUT  = 0.2


# ── LSTM network (mirrors services/ml_predictor._CropLSTM) ────────────────────

class _CropLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, num_crops: int) -> None:
        super().__init__()
        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=_DEFAULT_DROPOUT if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(_DEFAULT_DROPOUT)
        self.fc      = nn.Linear(hidden_size, num_crops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(self.dropout(h_n[-1]))


# ── Data shaping helpers ─────────────────────────────────────────────────────

def _build_yield_pivot(yield_long: pd.DataFrame, crop_index: dict[str, int]) -> pd.DataFrame:
    """Build wide yield matrix aligned to crop label order."""
    crops = list(crop_index.keys())
    subset = yield_long[yield_long["crop_name"].isin(crops)].copy()
    if subset.empty:
        raise ValueError("No yield rows for crops in crop_index.")

    pivot = (
        subset.pivot_table(index="time", columns="crop_name", values="yield_kg_ha", aggfunc="mean")
        .fillna(0.0)
        .sort_index()
    )
    ordered = sorted(crops, key=lambda c: crop_index[c])
    return pivot.reindex(columns=ordered, fill_value=0.0)


def _build_exog_matrix(exog_by_time: pd.DataFrame, index: pd.Index) -> pd.DataFrame:
    """Align canonical exogenous features to target index."""
    exog = exog_by_time.copy().set_index("time")[FEATURE_COLUMNS].sort_index()
    return exog.reindex(index).ffill().bfill().fillna(0.0)


# ── Sequence building ─────────────────────────────────────────────────────────

def _build_sequences(
    yield_pivot: pd.DataFrame,
    exog_matrix: pd.DataFrame,
    scaler: StandardScaler,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build overlapping input sequences and target labels for LSTM training.

    Logic Flow:
        For each timestep t ≥ seq_len:
            X[i] = scaled weather features for timesteps [t-seq_len : t]
            y[i] = normalised yield vector at timestep t
        Returns X of shape (N, seq_len, n_features) and
                y of shape (N, num_crops).

    Args:
        yield_pivot:      Wide yield DataFrame (index=time, cols=crop).
        weather_national: National monthly weather DataFrame.
        scaler:           Fitted StandardScaler (from sarimax artifacts).
        seq_len:          Number of past seasons to use as context.

    Returns:
        Tuple (X, y) as numpy float32 arrays.
    """
    # Scale canonical features with shared scaler fitted in SARIMAX training.
    weather_vals = scaler.transform(exog_matrix[FEATURE_COLUMNS].values.astype(np.float32)).astype(np.float32)

    # Build sequences
    X_list, y_list = [], []
    y_vals = yield_pivot.values.astype(np.float32)

    # Normalise yield targets to [0, 1] per crop
    y_max = np.maximum(y_vals.max(axis=0), 1.0)
    y_norm = y_vals / y_max

    for t in range(seq_len, len(weather_vals)):
        X_list.append(weather_vals[t - seq_len: t])
        y_list.append(y_norm[t])

    if not X_list:
        raise ValueError(
            f"Not enough data to build sequences (need >{seq_len} timesteps)."
        )

    return np.stack(X_list).astype(np.float32), np.stack(y_list).astype(np.float32)


# ── Training loop ─────────────────────────────────────────────────────────────

def _train(
    X: np.ndarray,
    y: np.ndarray,
    num_crops: int,
    hidden_size: int,
    num_layers: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> _CropLSTM:
    """Train the LSTM and return the fitted model.

    Logic Flow:
        Splits data 80/20 train/val.
        Uses MSELoss + Adam optimiser.
        Logs train/val loss every 5 epochs.
        Returns the model checkpoint with best validation loss.

    Args:
        X:           Input sequences (N, seq_len, n_features).
        y:           Target yield vectors (N, num_crops).
        num_crops:   Number of output classes.
        hidden_size: LSTM hidden dimension.
        num_layers:  Number of LSTM layers.
        epochs:      Training epochs.
        lr:          Learning rate.
        batch_size:  Mini-batch size.

    Returns:
        Trained _CropLSTM model in eval mode.
    """
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)

    n_features = X.shape[2]
    model = _CropLSTM(n_features, hidden_size, num_layers, num_crops)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state: dict = {}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimiser.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= max(len(val_ds), 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            logger.info(
                "lstm.epoch",
                epoch=epoch,
                epochs=epochs,
                train_loss=round(train_loss, 6),
                val_loss=round(val_loss, 6),
            )

    model.load_state_dict(best_state)
    model.eval()
    logger.info("lstm.train.complete", best_val_loss=round(best_val_loss, 6))
    return model


# ── Serialisation ─────────────────────────────────────────────────────────────

def _save_lstm_artifacts(
    region_code: str,
    model: _CropLSTM,
    hidden_size: int,
    num_layers: int,
    seq_len: int,
    n_features: int,
) -> None:
    """Save LSTM state_dict and config to ml/artifacts/{region_code}/.

    Args:
        region_code: Region identifier.
        model:       Trained LSTM model.
        hidden_size: LSTM hidden dimension.
        num_layers:  Number of LSTM layers.
        seq_len:     Training sequence length.
        n_features:  Input feature count.
    """
    art_dir = _ARTIFACTS_ROOT / region_code
    art_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), art_dir / "lstm_model.pt")
    (art_dir / "lstm_config.json").write_text(json.dumps({
        "input_size":  n_features,
        "hidden_size": hidden_size,
        "num_layers":  num_layers,
        "seq_len":     seq_len,
        "feature_columns": FEATURE_COLUMNS,
        "trained_at":  pd.Timestamp.utcnow().isoformat(),
    }, indent=2))
    logger.info("lstm.artifacts.saved", dir=str(art_dir))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    region_code: str,
    epochs: int,
    hidden_size: int,
    seq_len: int,
) -> None:
    """End-to-end LSTM training pipeline.

    Logic Flow:
        1. Load crop_index from existing SARIMAX artifacts (required).
        2. Load yield pivot and hex weather from TimescaleDB.
        3. Build overlapping input sequences.
        4. Train LSTM; save state_dict + config.

    Args:
        region_code: Runtime region identifier (e.g. 'IN').
        epochs:      Training epochs.
        hidden_size: LSTM hidden units.
        seq_len:     Number of past seasons per sequence.
    """
    logger.info("lstm.train.start", region_code=region_code)

    art_dir = _ARTIFACTS_ROOT / region_code
    crop_index_path = art_dir / "crop_index.json"
    scaler_path     = art_dir / "scaler.pkl"

    if not crop_index_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"SARIMAX artifacts not found at {art_dir}. "
            "Run ml/train/train_sarimax.py first."
        )

    crop_index: dict[str, int] = json.loads(crop_index_path.read_text())
    with scaler_path.open("rb") as f:
        scaler: StandardScaler = pickle.load(f)  # noqa: S301

    bundle = await build_training_bundle(region_code)
    yield_pivot = _build_yield_pivot(bundle.yield_long, crop_index)
    exog_matrix = _build_exog_matrix(bundle.exog_by_time, yield_pivot.index)

    X, y = _build_sequences(yield_pivot, exog_matrix, scaler, seq_len)
    num_crops  = y.shape[1]
    n_features = X.shape[2]

    logger.info(
        "lstm.data.ready",
        sequences=len(X),
        seq_len=seq_len,
        n_features=n_features,
        num_crops=num_crops,
    )

    model = _train(
        X, y,
        num_crops=num_crops,
        hidden_size=hidden_size,
        num_layers=_DEFAULT_LAYERS,
        epochs=epochs,
        lr=_DEFAULT_LR,
        batch_size=_DEFAULT_BATCH,
    )

    _save_lstm_artifacts(region_code, model, hidden_size, _DEFAULT_LAYERS, seq_len, n_features)
    logger.info("lstm.train.done", region_code=region_code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM crop prediction model.")
    parser.add_argument("--region",  required=True,                    help="Region code, e.g. IN")
    parser.add_argument("--epochs",  type=int, default=_DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--hidden",  type=int, default=_DEFAULT_HIDDEN, help="LSTM hidden size")
    parser.add_argument("--seq-len", type=int, default=_DEFAULT_SEQ_LEN, help="Sequence length (seasons)")
    args = parser.parse_args()

    asyncio.run(main(args.region, args.epochs, args.hidden, args.seq_len))
