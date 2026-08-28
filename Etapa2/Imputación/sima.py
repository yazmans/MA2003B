"""SIMA: imputación multivariada de datos horarios de calidad del aire.

Este archivo integra en un único programa la versión efectiva de
``notebooks/Final_SIMA.ipynb`` y la convierte en una herramienta reproducible.

El modelo conserva las dos ideas centrales del notebook:

1. Un Transformer produce una reconstrucción estructural de la ventana.
2. Un denoiser condicional añade escenarios residuales para cuantificar
   incertidumbre.

Importante: las clases originales se llamaron ``SAITS_Base`` y ``CSDI_Base``,
pero son aproximaciones educativas, no reproducciones completas de los
algoritmos publicados. La guía de estudiantes explica las diferencias.

Ejemplos:
    uv run python sima.py demo
    uv run python sima.py inspect --input data/df_simanew_cleaned.csv
    uv run python sima.py run --input data/df_simanew_cleaned.csv --epochs 50
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

LOGGER = logging.getLogger("sima")
MODEL_FORMAT_VERSION = 1

# El orden es parte del contrato del checkpoint: no debe cambiarse después de
# entrenar, porque cada posición corresponde a una salida distinta de la red.
FEATURE_COLUMNS: tuple[str, ...] = (
    "CO",
    "NO",
    "NO2",
    "NOX",
    "O3",
    "PM10",
    "PM2.5",
    "PRS",
    "RAINF",
    "RH",
    "SO2",
    "SR",
    "TOUT",
    "WSR",
    "WDR",
)


class SIMAError(RuntimeError):
    """Error esperado con un mensaje útil para quien ejecuta el programa."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Hiperparámetros que determinan la forma del modelo."""

    seq_len: int = 168
    n_features: int = len(FEATURE_COLUMNS)
    hidden_size: int = 64
    n_heads: int = 4
    n_layers: int = 2
    diffusion_steps: int = 50
    dropout: float = 0.1
    residual_scale: float = 0.1

    def validate(self) -> None:
        if self.seq_len < 2:
            raise SIMAError("seq_len debe ser al menos 2.")
        if self.n_features != len(FEATURE_COLUMNS):
            raise SIMAError(
                f"El modelo espera {len(FEATURE_COLUMNS)} variables en el orden documentado."
            )
        if self.hidden_size % self.n_heads != 0:
            raise SIMAError("hidden_size debe ser divisible entre n_heads.")
        if self.diffusion_steps < 1:
            raise SIMAError("diffusion_steps debe ser positivo.")
        if not 0 <= self.dropout < 1:
            raise SIMAError("dropout debe estar en [0, 1).")


@dataclass(slots=True)
class StationSeries:
    """Serie horaria continua de una estación."""

    station_id: str
    time: pd.DatetimeIndex
    raw_values: np.ndarray
    values: np.ndarray | None = None


@dataclass(slots=True)
class DataBundle:
    """Datos normalizados junto con su normalizador y corte temporal."""

    stations: list[StationSeries]
    normalizer: NanMinMaxNormalizer
    train_ends: list[int]


class NanMinMaxNormalizer:
    """Min-Max global que ignora NaN y conserva su posición.

    Se implementa aquí, en vez de serializar un objeto de scikit-learn, para
    que el checkpoint solo contenga tensores y tipos básicos. El ajuste se hace
    exclusivamente con el tramo de entrenamiento para evitar fuga temporal.
    """

    def __init__(self, minimum: np.ndarray | None = None, span: np.ndarray | None = None):
        self.minimum = None if minimum is None else np.asarray(minimum, dtype=np.float32)
        self.span = None if span is None else np.asarray(span, dtype=np.float32)

    @property
    def is_fitted(self) -> bool:
        return self.minimum is not None and self.span is not None

    def fit(self, arrays: Sequence[np.ndarray], train_ends: Sequence[int]) -> NanMinMaxNormalizer:
        n_features = len(FEATURE_COLUMNS)
        minimum = np.full(n_features, np.inf, dtype=np.float64)
        maximum = np.full(n_features, -np.inf, dtype=np.float64)

        for values, end in zip(arrays, train_ends, strict=True):
            train = np.asarray(values[:end], dtype=np.float64)
            for feature in range(n_features):
                observed = train[:, feature]
                observed = observed[np.isfinite(observed)]
                if observed.size:
                    minimum[feature] = min(minimum[feature], float(observed.min()))
                    maximum[feature] = max(maximum[feature], float(observed.max()))

        missing = [
            FEATURE_COLUMNS[index]
            for index in range(n_features)
            if not np.isfinite(minimum[index]) or not np.isfinite(maximum[index])
        ]
        if missing:
            raise SIMAError(
                "No hay valores observados en entrenamiento para: " + ", ".join(missing)
            )

        span = maximum - minimum
        # Una variable constante sigue siendo válida; su representación normalizada es cero.
        span[span == 0] = 1.0
        self.minimum = minimum.astype(np.float32)
        self.span = span.astype(np.float32)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return ((values - self.minimum) / self.span).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return (values * self.span + self.minimum).astype(np.float32)

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        self._require_fitted()
        minimum = torch.as_tensor(self.minimum, dtype=values.dtype, device=values.device)
        span = torch.as_tensor(self.span, dtype=values.dtype, device=values.device)
        return values * span + minimum

    def state_dict(self) -> dict[str, torch.Tensor]:
        self._require_fitted()
        return {
            "minimum": torch.from_numpy(self.minimum.copy()),
            "span": torch.from_numpy(self.span.copy()),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> NanMinMaxNormalizer:
        return cls(state["minimum"].cpu().numpy(), state["span"].cpu().numpy())

    def _require_fitted(self) -> None:
        if not self.is_fitted:
            raise SIMAError("El normalizador todavía no ha sido ajustado.")


class AirQualityDataProcessor:
    """Valida el CSV, promedia duplicados y crea una cuadrícula horaria."""

    def __init__(self, csv_path: Path):
        self.csv_path = csv_path

    def load_raw(self) -> list[StationSeries]:
        if not self.csv_path.is_file():
            raise SIMAError(
                f"No se encontró el CSV: {self.csv_path}. Consulta data/README.md."
            )

        header = pd.read_csv(self.csv_path, nrows=0)
        required = {"ID", "time", *FEATURE_COLUMNS}
        missing = sorted(required - set(header.columns))
        if missing:
            raise SIMAError("Faltan columnas obligatorias: " + ", ".join(missing))

        LOGGER.info("Cargando %s", self.csv_path)
        frame = pd.read_csv(self.csv_path, usecols=["ID", "time", *FEATURE_COLUMNS])
        frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
        invalid_time = int(frame["time"].isna().sum())
        if invalid_time:
            LOGGER.warning("Se descartaron %d filas con fecha inválida.", invalid_time)
            frame = frame.dropna(subset=["time"])
        if frame.empty:
            raise SIMAError("El CSV no contiene filas con fecha válida.")

        frame[list(FEATURE_COLUMNS)] = frame[list(FEATURE_COLUMNS)].apply(
            pd.to_numeric, errors="coerce"
        )

        stations: list[StationSeries] = []
        for station_id in pd.unique(frame["ID"]):
            station_frame = frame.loc[frame["ID"] == station_id, ["time", *FEATURE_COLUMNS]]
            # Promediar duplicados evita elegir arbitrariamente una lectura cuando hay
            # más de un registro para la misma estación y hora.
            hourly = station_frame.groupby("time", sort=True)[list(FEATURE_COLUMNS)].mean()
            if hourly.empty:
                continue
            full_time = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
            hourly = hourly.reindex(full_time)
            # Se conserva float64 para poder devolver cada lectura original sin
            # pérdida decimal. Solo la copia normalizada que entra a la red usa
            # float32 para ahorrar memoria y acelerar PyTorch.
            raw_values = hourly.to_numpy(dtype=np.float64, copy=True)
            stations.append(
                StationSeries(str(station_id), pd.DatetimeIndex(full_time), raw_values)
            )

        if not stations:
            raise SIMAError("No fue posible construir ninguna serie por estación.")
        LOGGER.info("Se construyeron %d series horarias continuas.", len(stations))
        return stations

    def prepare(
        self,
        seq_len: int,
        train_fraction: float = 0.8,
        normalizer: NanMinMaxNormalizer | None = None,
    ) -> DataBundle:
        if not 0.5 <= train_fraction < 1:
            raise SIMAError("train_fraction debe estar en [0.5, 1).")

        stations = self.load_raw()
        train_ends = [temporal_train_end(len(s.time), seq_len, train_fraction) for s in stations]
        scaler = normalizer or NanMinMaxNormalizer()
        if not scaler.is_fitted:
            scaler.fit([s.raw_values for s in stations], train_ends)
        for station in stations:
            station.values = scaler.transform(station.raw_values)
        return DataBundle(stations, scaler, train_ends)


def temporal_train_end(n_rows: int, seq_len: int, train_fraction: float) -> int:
    """Corte cronológico que reserva al menos una ventana para validación."""

    if n_rows < seq_len:
        return n_rows
    if n_rows < 2 * seq_len:
        return n_rows
    proposed = int(math.floor(n_rows * train_fraction))
    return min(max(seq_len, proposed), n_rows - seq_len)


def covering_window_starts(
    n_rows: int,
    seq_len: int,
    stride: int,
    start: int = 0,
    stop: int | None = None,
) -> list[int]:
    """Inicios de ventana que incluyen ambos bordes del intervalo.

    El notebook dejaba algunas colas sin predicción cuando la longitud no era
    múltiplo del stride. Añadir explícitamente la última ventana evita rellenar
    esos puntos accidentalmente con el mínimo del normalizador.
    """

    if stride < 1:
        raise SIMAError("stride debe ser positivo.")
    stop = n_rows if stop is None else min(stop, n_rows)
    length = stop - start
    if length < seq_len:
        return []
    last = stop - seq_len
    starts = list(range(start, last + 1, stride))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts


def make_window_references(
    bundle: DataBundle,
    seq_len: int,
    stride: int,
    min_observed: float,
    split: str,
) -> list[tuple[int, int]]:
    """Crea referencias sin copiar las ventanas completas a memoria."""

    if not 0 <= min_observed <= 1:
        raise SIMAError("min_observed debe estar en [0, 1].")
    if split not in {"train", "validation", "all"}:
        raise SIMAError(f"Split desconocido: {split}")

    references: list[tuple[int, int]] = []
    for station_index, (station, train_end) in enumerate(
        zip(bundle.stations, bundle.train_ends, strict=True)
    ):
        assert station.values is not None
        if split == "train":
            starts = covering_window_starts(len(station.time), seq_len, stride, 0, train_end)
        elif split == "validation":
            starts = covering_window_starts(
                len(station.time), seq_len, stride, train_end, len(station.time)
            )
        else:
            starts = covering_window_starts(len(station.time), seq_len, stride)

        for window_start in starts:
            window = station.values[window_start : window_start + seq_len]
            observed_ratio = float(np.isfinite(window).mean())
            if observed_ratio >= min_observed:
                references.append((station_index, window_start))
    return references


class LazyAirQualityDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset perezoso equivalente al notebook, con metadatos de estación."""

    def __init__(
        self,
        stations: Sequence[StationSeries],
        references: Sequence[tuple[int, int]],
        seq_len: int,
    ):
        self.stations = stations
        self.references = list(references)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        station_index, start = self.references[index]
        values = self.stations[station_index].values
        assert values is not None
        window = values[start : start + self.seq_len]
        # La máscara se calcula desde NaN. Un cero es una observación válida y no
        # debe confundirse con un dato faltante, como ocurría en la imputación final.
        observed_mask = np.isfinite(window).astype(np.float32)
        observed_data = np.nan_to_num(window, nan=0.0, copy=True).astype(np.float32)
        return {
            "observed_data": torch.from_numpy(observed_data),
            "observed_mask": torch.from_numpy(observed_mask),
            "station_index": torch.tensor(station_index, dtype=torch.long),
            "start": torch.tensor(start, dtype=torch.long),
        }


class SAITSBase(nn.Module):
    """Reconstrucción estructural inspirada en SAITS.

    A diferencia de la celda original, se añade una codificación de posición:
    sin ella un Transformer no puede distinguir la primera hora de la última.
    Para reproducir el artículo SAITS aún faltarían sus dos bloques DMSA y la
    combinación ponderada; esta diferencia se documenta explícitamente.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_projection = nn.Linear(config.n_features * 2, config.hidden_size)
        self.position_embedding = nn.Embedding(config.seq_len, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.n_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.output_projection = nn.Linear(config.hidden_size, config.n_features)

    def forward(
        self, values: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        time_steps = values.shape[1]
        positions = torch.arange(time_steps, device=values.device)
        combined = torch.cat([values, mask], dim=-1)
        embedded = self.input_projection(combined) + self.position_embedding(positions)[None, :, :]
        raw_prediction = self.output_projection(self.transformer(embedded))
        filled = values * mask + raw_prediction * (1.0 - mask)
        return filled, raw_prediction


class CSDIBase(nn.Module):
    """Denoiser residual condicional inspirado en CSDI.

    Se usa el paso de ruido que en el notebook estaba declarado pero no conectado.
    Esto sigue siendo un denoiser de una etapa, no el proceso iterativo completo de
    difusión de CSDI.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.step_embedding = nn.Embedding(config.diffusion_steps, config.hidden_size)
        self.condition_projection = nn.Linear(config.n_features, config.hidden_size)
        self.residual_layers = nn.Sequential(
            nn.Linear(config.n_features + config.hidden_size, 128),
            nn.GELU(),
            nn.Linear(128, config.n_features),
        )

    def forward(
        self,
        noisy_values: torch.Tensor,
        diffusion_step: torch.Tensor,
        structural_condition: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.condition_projection(structural_condition)
        step = self.step_embedding(diffusion_step)[:, None, :]
        return self.residual_layers(torch.cat([noisy_values, condition + step], dim=-1))


class SIMAHybrid(nn.Module):
    """Cascada estructural + denoiser condicional."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.saits = SAITSBase(config)
        self.csdi = CSDIBase(config)


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SIMAError("Se solicitó CUDA, pero PyTorch no detecta una GPU compatible.")
    return torch.device(requested)


def random_hidden_mask(observed_mask: torch.Tensor, ratio: float) -> torch.Tensor:
    """Oculta aleatoriamente solo celdas cuyo valor real sí conocemos."""

    if not 0 < ratio < 1:
        raise SIMAError("mask_ratio debe estar en (0, 1).")
    return (torch.rand_like(observed_mask) < ratio).float() * observed_mask


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (((prediction - target) ** 2) * mask).sum() / denominator


def make_loader(
    bundle: DataBundle,
    references: Sequence[tuple[int, int]],
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader[dict[str, torch.Tensor]]:
    if not references:
        raise SIMAError("No hay ventanas válidas para esta partición.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        LazyAirQualityDataset(bundle.stations, references, seq_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def quick_validation_loss(
    model: SIMAHybrid,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    mask_ratio: float,
    max_batches: int = 10,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            values = batch["observed_data"].to(device)
            observed = batch["observed_mask"].to(device)
            hidden = random_hidden_mask(observed, mask_ratio)
            input_mask = observed - hidden
            _, raw_prediction = model.saits(values * input_mask, input_mask)
            losses.append(float(masked_mse(raw_prediction, values, hidden).item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_model(
    bundle: DataBundle,
    config: ModelConfig,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    mask_ratio: float,
    train_stride: int,
    eval_stride: int,
    min_observed: float,
    workers: int,
    seed: int,
) -> tuple[SIMAHybrid, list[dict[str, float]]]:
    """Entrena con huecos artificiales y validación cronológica separada."""

    train_refs = make_window_references(
        bundle, config.seq_len, train_stride, min_observed, "train"
    )
    validation_refs = make_window_references(
        bundle, config.seq_len, eval_stride, min_observed, "validation"
    )
    LOGGER.info(
        "Ventanas: %d entrenamiento, %d validación (corte temporal).",
        len(train_refs),
        len(validation_refs),
    )
    train_loader = make_loader(
        bundle, train_refs, config.seq_len, batch_size, True, workers, seed, device
    )
    validation_loader = make_loader(
        bundle, validation_refs, config.seq_len, batch_size, False, workers, seed, device
    )

    model = SIMAHybrid(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_reconstruction = 0.0
        total_denoising = 0.0
        n_batches = 0

        for batch in train_loader:
            values = batch["observed_data"].to(device, non_blocking=True)
            observed = batch["observed_mask"].to(device, non_blocking=True)
            hidden = random_hidden_mask(observed, mask_ratio)
            input_mask = observed - hidden
            input_values = values * input_mask

            optimizer.zero_grad(set_to_none=True)
            coarse, raw_prediction = model.saits(input_values, input_mask)
            reconstruction = masked_mse(raw_prediction, values, hidden)

            noise = torch.randn_like(values)
            steps = torch.randint(0, config.diffusion_steps, (values.shape[0],), device=device)
            predicted_noise = model.csdi(noise, steps, coarse)
            denoising = F.mse_loss(predicted_noise, noise)
            loss = reconstruction + denoising
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_reconstruction += float(reconstruction.item())
            total_denoising += float(denoising.item())
            n_batches += 1

        if not n_batches:
            raise SIMAError("El entrenamiento no produjo ningún lote.")
        validation = quick_validation_loss(
            model, validation_loader, device, mask_ratio, max_batches=10
        )
        row = {
            "epoch": float(epoch),
            "loss": total_loss / n_batches,
            "reconstruction_loss": total_reconstruction / n_batches,
            "denoising_loss": total_denoising / n_batches,
            "validation_masked_mse": validation,
        }
        history.append(row)
        LOGGER.info(
            "Época %d/%d | loss %.5f | reconstrucción %.5f | validación %.5f",
            epoch,
            epochs,
            row["loss"],
            row["reconstruction_loss"],
            row["validation_masked_mse"],
        )
    return model, history


def save_checkpoint(
    path: Path,
    model: SIMAHybrid,
    normalizer: NanMinMaxNormalizer,
    history: Sequence[dict[str, float]],
    train_fraction: float,
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": MODEL_FORMAT_VERSION,
        "model_config": asdict(model.config),
        "feature_columns": list(FEATURE_COLUMNS),
        "model_state": model.state_dict(),
        "normalizer": normalizer.state_dict(),
        "training": {"train_fraction": train_fraction, "seed": seed},
        "history": list(history),
    }
    torch.save(payload, path)
    LOGGER.info("Checkpoint guardado en %s", path)


def load_checkpoint(
    path: Path, device: torch.device
) -> tuple[SIMAHybrid, NanMinMaxNormalizer, dict[str, Any]]:
    if not path.is_file():
        raise SIMAError(f"No se encontró el checkpoint: {path}")
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibilidad con versiones antiguas de PyTorch.
        payload = torch.load(path, map_location=device)
    if payload.get("format_version") != MODEL_FORMAT_VERSION:
        raise SIMAError("La versión del checkpoint no es compatible con este script.")
    if tuple(payload.get("feature_columns", ())) != FEATURE_COLUMNS:
        raise SIMAError("El orden de variables del checkpoint no coincide con el script.")
    config = ModelConfig(**payload["model_config"])
    config.validate()
    model = SIMAHybrid(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    normalizer = NanMinMaxNormalizer.from_state_dict(payload["normalizer"])
    return model, normalizer, payload


def sample_imputations(
    model: SIMAHybrid,
    values: torch.Tensor,
    observed_mask: torch.Tensor,
    n_samples: int,
) -> torch.Tensor:
    """Genera escenarios y preserva exactamente cada celda observada."""

    if n_samples < 1:
        raise SIMAError("n_samples debe ser positivo.")
    coarse, _ = model.saits(values * observed_mask, observed_mask)
    samples: list[torch.Tensor] = []
    for _ in range(n_samples):
        noise = torch.randn_like(values)
        steps = torch.randint(
            0, model.config.diffusion_steps, (values.shape[0],), device=values.device
        )
        predicted_noise = model.csdi(noise, steps, coarse)
        candidate = coarse + (noise - predicted_noise) * model.config.residual_scale
        samples.append(values * observed_mask + candidate * (1.0 - observed_mask))
    return torch.stack(samples, dim=0)


def basic_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """MAE, RMSE, MRE y R² sin depender de otra biblioteca."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(target) & np.isfinite(prediction)
    target = target[valid]
    prediction = prediction[valid]
    if not target.size:
        return {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mre_percent": float("nan"),
            "r2": float("nan"),
        }

    errors = prediction - target
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    relative = np.abs(target) > 1.0
    mre = (
        float(np.mean(np.abs(errors[relative]) / np.abs(target[relative])) * 100)
        if relative.any()
        else float("nan")
    )
    denominator = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - float(np.sum(errors**2)) / denominator if denominator > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "mre_percent": mre, "r2": r2}


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.size < 2 or np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def quantile_loss(target: np.ndarray, ensemble: np.ndarray, quantile: float = 0.95) -> float:
    """Pinball loss del cuantil indicado; ensemble tiene forma (muestras, puntos)."""

    predicted_quantile = np.quantile(ensemble, quantile, axis=0)
    error = target - predicted_quantile
    return float(np.mean(np.maximum((quantile - 1.0) * error, quantile * error)))


def ensemble_crps(target: np.ndarray, ensemble: np.ndarray) -> float:
    """CRPS empírico completo para un ensamble equiprobable.

    CRPS = E|X-y| - 0.5 E|X-X'|. La celda original calculaba solo el
    primer término, que es un MAE probabilístico y no el CRPS completo.
    """

    n_samples = ensemble.shape[0]
    first_term = float(np.mean(np.abs(ensemble - target[None, :])))
    if n_samples == 1:
        return first_term
    ordered = np.sort(ensemble, axis=0)
    coefficients = 2 * np.arange(1, n_samples + 1) - n_samples - 1
    half_pairwise = np.mean(
        np.sum(coefficients[:, None] * ordered, axis=0) / (n_samples**2)
    )
    return float(first_term - half_pairwise)


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def plot_evaluation_example(
    path: Path,
    target: np.ndarray,
    hidden_mask: np.ndarray,
    ensemble: np.ndarray,
    feature_index: int,
) -> None:
    """Guarda un ejemplo sin abrir una ventana gráfica (funciona en servidores)."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = np.arange(target.shape[0])
    mean = ensemble.mean(axis=0)
    lower = np.quantile(ensemble, 0.05, axis=0)
    upper = np.quantile(ensemble, 0.95, axis=0)
    hidden = hidden_mask[:, feature_index].astype(bool)

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(time, target[:, feature_index], "k.-", alpha=0.55, label="Valor real")
    axis.plot(time, mean[:, feature_index], color="tab:red", label="Imputación media")
    axis.fill_between(
        time,
        lower[:, feature_index],
        upper[:, feature_index],
        color="tab:red",
        alpha=0.2,
        label="Intervalo empírico 90%",
    )
    axis.scatter(
        time[hidden],
        target[hidden, feature_index],
        color="tab:blue",
        marker="x",
        label="Dato ocultado para evaluar",
    )
    axis.set_title(f"Auditoría de imputación: {FEATURE_COLUMNS[feature_index]}")
    axis.set_xlabel("Hora dentro de la ventana")
    axis.set_ylabel("Escala física de la variable")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def evaluate_model(
    model: SIMAHybrid,
    bundle: DataBundle,
    device: torch.device,
    output_dir: Path,
    batch_size: int,
    eval_stride: int,
    min_observed: float,
    mask_ratio: float,
    n_samples: int,
    max_batches: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Audita solo valores conocidos que se ocultaron artificialmente.

    Esta es la diferencia metodológica más importante respecto a varias celdas
    del notebook: comparar la salida con las entradas visibles produce métricas
    artificialmente perfectas porque el modelo copia esos valores por diseño.
    """

    references = make_window_references(
        bundle,
        model.config.seq_len,
        eval_stride,
        min_observed,
        "validation",
    )
    loader = make_loader(
        bundle,
        references,
        model.config.seq_len,
        batch_size,
        False,
        workers,
        seed,
        device,
    )
    model.eval()

    true_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    ensemble_parts: list[np.ndarray] = []
    feature_true: dict[int, list[np.ndarray]] = defaultdict(list)
    feature_prediction: dict[int, list[np.ndarray]] = defaultdict(list)
    station_true: dict[int, list[np.ndarray]] = defaultdict(list)
    station_prediction: dict[int, list[np.ndarray]] = defaultdict(list)
    station_domain: dict[int, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pm10_parts: list[np.ndarray] = []
    pm25_parts: list[np.ndarray] = []
    nox_parts: list[np.ndarray] = []
    o3_parts: list[np.ndarray] = []
    persistence_model_sse = 0.0
    persistence_baseline_sse = 0.0
    example: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    index_pm10 = FEATURE_COLUMNS.index("PM10")
    index_pm25 = FEATURE_COLUMNS.index("PM2.5")
    index_nox = FEATURE_COLUMNS.index("NOX")
    index_o3 = FEATURE_COLUMNS.index("O3")

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            values = batch["observed_data"].to(device)
            original_mask = batch["observed_mask"].to(device)
            hidden = random_hidden_mask(original_mask, mask_ratio)
            input_mask = original_mask - hidden
            samples_norm = sample_imputations(model, values, input_mask, n_samples)
            target_phys = bundle.normalizer.inverse_tensor(values)
            samples_phys = bundle.normalizer.inverse_tensor(samples_norm)
            prediction_phys = samples_phys.mean(dim=0)

            hidden_bool = hidden.bool()
            target_np = target_phys.cpu().numpy()
            prediction_np = prediction_phys.cpu().numpy()
            ensemble_np = samples_phys.cpu().numpy()
            hidden_np = hidden_bool.cpu().numpy()

            true_parts.append(target_np[hidden_np])
            prediction_parts.append(prediction_np[hidden_np])
            ensemble_parts.append(ensemble_np[:, hidden_np])

            for feature_index in range(len(FEATURE_COLUMNS)):
                feature_mask = hidden_np[:, :, feature_index]
                if feature_mask.any():
                    feature_true[feature_index].append(
                        target_np[:, :, feature_index][feature_mask]
                    )
                    feature_prediction[feature_index].append(
                        prediction_np[:, :, feature_index][feature_mask]
                    )

            station_indices = batch["station_index"].numpy()
            for row, station_index_raw in enumerate(station_indices):
                station_index = int(station_index_raw)
                row_mask = hidden_np[row]
                if row_mask.any():
                    station_true[station_index].append(target_np[row][row_mask])
                    station_prediction[station_index].append(prediction_np[row][row_mask])
                domain = station_domain[station_index]
                domain["pm10"].append(prediction_np[row, :, index_pm10])
                domain["pm25"].append(prediction_np[row, :, index_pm25])
                domain["nox"].append(prediction_np[row, :, index_nox])
                domain["o3"].append(prediction_np[row, :, index_o3])

            pm10_parts.append(prediction_np[:, :, index_pm10].ravel())
            pm25_parts.append(prediction_np[:, :, index_pm25].ravel())
            nox_parts.append(prediction_np[:, :, index_nox].ravel())
            o3_parts.append(prediction_np[:, :, index_o3].ravel())

            # Persistencia: para cada punto ocultado en t, usa el valor observado
            # 24 horas antes. Solo se comparan pares donde ambos valores existen.
            if model.config.seq_len > 24:
                persistence_mask = hidden_bool[:, 24:, :] & original_mask[:, :-24, :].bool()
                if persistence_mask.any():
                    target_future = target_phys[:, 24:, :][persistence_mask]
                    model_future = prediction_phys[:, 24:, :][persistence_mask]
                    baseline = target_phys[:, :-24, :][persistence_mask]
                    persistence_model_sse += float(torch.sum((target_future - model_future) ** 2))
                    persistence_baseline_sse += float(torch.sum((target_future - baseline) ** 2))

            if example is None:
                example = (target_np[0], hidden_np[0], ensemble_np[:, 0])

    if not true_parts:
        raise SIMAError("No hubo valores observados para ocultar durante la evaluación.")

    y_true = np.concatenate(true_parts)
    y_prediction = np.concatenate(prediction_parts)
    ensemble = np.concatenate(ensemble_parts, axis=1)
    aggregate = basic_metrics(y_true, y_prediction)
    feature_rows: list[dict[str, Any]] = []
    for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
        if feature_index not in feature_true:
            continue
        feature_target = np.concatenate(feature_true[feature_index])
        feature_pred = np.concatenate(feature_prediction[feature_index])
        feature_rows.append(
            {
                "feature": feature_name,
                "n_hidden": int(feature_target.size),
                **basic_metrics(feature_target, feature_pred),
                "pearson": safe_pearson(feature_target, feature_pred),
            }
        )

    station_rows: list[dict[str, Any]] = []
    for station_index, station in enumerate(bundle.stations):
        if station_index not in station_true:
            continue
        station_target = np.concatenate(station_true[station_index])
        station_pred = np.concatenate(station_prediction[station_index])
        domain = station_domain[station_index]
        station_pm10 = np.concatenate(domain["pm10"])
        station_pm25 = np.concatenate(domain["pm25"])
        station_rows.append(
            {
                "station_id": station.station_id,
                "n_hidden": int(station_target.size),
                **basic_metrics(station_target, station_pred),
                "pm25_gt_pm10_percent": float(np.mean(station_pm25 > station_pm10) * 100),
                "nox_o3_pearson": safe_pearson(
                    np.concatenate(domain["nox"]), np.concatenate(domain["o3"])
                ),
            }
        )

    pm10 = np.concatenate(pm10_parts)
    pm25 = np.concatenate(pm25_parts)
    nox = np.concatenate(nox_parts)
    o3 = np.concatenate(o3_parts)
    skill_score = (
        1.0 - persistence_model_sse / persistence_baseline_sse
        if persistence_baseline_sse > 0
        else float("nan")
    )
    summary: dict[str, Any] = {
        "protocol": {
            "split": "chronological_holdout",
            "artificial_mask_ratio": mask_ratio,
            "n_ensemble_samples": n_samples,
            "n_evaluated_values": int(y_true.size),
            "max_batches": max_batches or None,
        },
        "aggregate_metrics_physical_scale": {
            key: finite_or_none(value) for key, value in aggregate.items()
        },
        "probabilistic_metrics_physical_scale": {
            "quantile_loss_95": finite_or_none(quantile_loss(y_true, ensemble, 0.95)),
            "crps": finite_or_none(ensemble_crps(y_true, ensemble)),
        },
        "domain_metrics": {
            "pm25_gt_pm10_percent": finite_or_none(float(np.mean(pm25 > pm10) * 100)),
            "nox_o3_pearson": finite_or_none(safe_pearson(nox, o3)),
            "mean_feature_pearson": finite_or_none(
                float(np.nanmean([row["pearson"] for row in feature_rows]))
            ),
            "skill_score_vs_24h_persistence": finite_or_none(skill_score),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(feature_rows).to_csv(output_dir / "metrics_by_feature.csv", index=False)
    pd.DataFrame(station_rows).to_csv(output_dir / "metrics_by_station.csv", index=False)
    if example is not None:
        plot_evaluation_example(
            output_dir / "imputation_example_pm25.png",
            example[0],
            example[1],
            example[2],
            index_pm25,
        )
    LOGGER.info("Evaluación guardada en %s", output_dir)
    return summary


def batched(items: Sequence[int], size: int) -> Iterable[Sequence[int]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def impute_station(
    model: SIMAHybrid,
    station: StationSeries,
    normalizer: NanMinMaxNormalizer,
    device: torch.device,
    stride: int,
    batch_size: int,
    n_samples: int,
) -> np.ndarray:
    """Imputa una estación con ventanas solapadas y preserva observaciones."""

    assert station.values is not None
    normalized = station.values
    n_rows, n_features = normalized.shape
    seq_len = model.config.seq_len

    if n_rows < seq_len:
        starts = [0]
    else:
        starts = covering_window_starts(n_rows, seq_len, stride)

    aggregate = np.zeros((n_rows, n_features), dtype=np.float64)
    counts = np.zeros((n_rows, n_features), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start_batch in batched(starts, batch_size):
            batch_values: list[np.ndarray] = []
            batch_masks: list[np.ndarray] = []
            valid_lengths: list[int] = []
            for start in start_batch:
                window = normalized[start : start + seq_len]
                valid_length = len(window)
                padded = np.full((seq_len, n_features), np.nan, dtype=np.float32)
                padded[:valid_length] = window
                batch_masks.append(np.isfinite(padded).astype(np.float32))
                batch_values.append(np.nan_to_num(padded, nan=0.0))
                valid_lengths.append(valid_length)

            values_tensor = torch.from_numpy(np.stack(batch_values)).to(device)
            mask_tensor = torch.from_numpy(np.stack(batch_masks)).to(device)
            predictions = sample_imputations(
                model, values_tensor, mask_tensor, n_samples
            ).mean(dim=0).cpu().numpy()

            locations = zip(start_batch, valid_lengths, strict=True)
            for row, (start, valid_length) in enumerate(locations):
                end = start + valid_length
                aggregate[start:end] += predictions[row, :valid_length]
                counts[start:end] += 1.0

    if np.any(counts == 0):
        raise SIMAError(f"La estación {station.station_id} quedó con puntos sin cubrir.")
    imputed_normalized = (aggregate / counts).astype(np.float32)
    imputed_physical = normalizer.inverse_transform(imputed_normalized)
    # Partir de la matriz float64 original evita incluso el redondeo que se
    # produciría al asignar una observación a un contenedor float32.
    completed = station.raw_values.copy()
    missing = ~np.isfinite(completed)
    completed[missing] = imputed_physical[missing]
    return completed


def impute_dataset(
    model: SIMAHybrid,
    bundle: DataBundle,
    device: torch.device,
    output_csv: Path,
    stride: int,
    batch_size: int,
    n_samples: int,
) -> tuple[Path, Path]:
    frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for station in bundle.stations:
        LOGGER.info("Imputando estación %s (%d horas).", station.station_id, len(station.time))
        imputed = impute_station(
            model,
            station,
            bundle.normalizer,
            device,
            stride,
            batch_size,
            n_samples,
        )
        frame = pd.DataFrame(imputed, columns=FEATURE_COLUMNS)
        frame.insert(0, "time", station.time)
        frame.insert(0, "ID", station.station_id)
        frames.append(frame)
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            missing_before = int(np.isnan(station.raw_values[:, feature_index]).sum())
            audit_rows.append(
                {
                    "station_id": station.station_id,
                    "feature": feature,
                    "missing_before": missing_before,
                    "missing_after": int(np.isnan(imputed[:, feature_index]).sum()),
                }
            )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(output_csv, index=False)
    audit_path = output_csv.with_name(f"{output_csv.stem}_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    LOGGER.info("Dataset imputado guardado en %s", output_csv)
    return output_csv, audit_path


def data_summary(stations: Sequence[StationSeries]) -> tuple[pd.DataFrame, pd.DataFrame]:
    station_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for station in stations:
        missing = np.isnan(station.raw_values)
        station_rows.append(
            {
                "station_id": station.station_id,
                "start": station.time.min(),
                "end": station.time.max(),
                "hours": len(station.time),
                "missing_percent_all": float(missing.mean() * 100),
            }
        )
        for feature_index, feature in enumerate(FEATURE_COLUMNS):
            missing_rows.append(
                {
                    "station_id": station.station_id,
                    "feature": feature,
                    "missing_percent": float(missing[:, feature_index].mean() * 100),
                }
            )
    return pd.DataFrame(station_rows), pd.DataFrame(missing_rows)


def generate_demo_csv(path: Path, hours: int, seed: int) -> Path:
    """Crea datos sintéticos plausibles; sirven para probar el flujo, no para investigar."""

    if hours < 96:
        raise SIMAError("La demostración necesita al menos 96 horas.")
    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    time = pd.date_range("2025-01-01", periods=hours, freq="h")
    hour = np.arange(hours)

    for station_index, station_id in enumerate(("DEMO_NORTE", "DEMO_SUR")):
        daily = np.sin(2 * np.pi * hour / 24 + station_index * 0.4)
        weekly = np.sin(2 * np.pi * hour / 168)
        pm10 = np.clip(55 + 18 * daily + 8 * weekly + rng.normal(0, 5, hours), 3, None)
        pm25 = np.clip(0.58 * pm10 + rng.normal(0, 3, hours), 1, pm10)
        nox = np.clip(32 + 12 * daily + rng.normal(0, 4, hours), 0, None)
        o3 = np.clip(45 - 0.55 * nox - 8 * daily + rng.normal(0, 3, hours), 0, None)
        values = {
            "CO": np.clip(0.7 + 0.2 * daily + rng.normal(0, 0.05, hours), 0, None),
            "NO": np.clip(nox * 0.35 + rng.normal(0, 1, hours), 0, None),
            "NO2": np.clip(nox * 0.65 + rng.normal(0, 1, hours), 0, None),
            "NOX": nox,
            "O3": o3,
            "PM10": pm10,
            "PM2.5": pm25,
            "PRS": 720 + 2 * weekly + rng.normal(0, 0.5, hours),
            "RAINF": np.where(rng.random(hours) < 0.04, rng.uniform(0.1, 8, hours), 0),
            "RH": np.clip(58 - 12 * daily + rng.normal(0, 4, hours), 15, 100),
            "SO2": np.clip(5 + rng.normal(0, 1.2, hours), 0, None),
            "SR": np.clip(0.7 * np.sin(np.pi * (hour % 24 - 6) / 12), 0, None),
            "TOUT": 23 + 7 * daily + rng.normal(0, 1, hours),
            "WSR": np.clip(8 + 3 * daily + rng.normal(0, 1.5, hours), 0, None),
            "WDR": (180 + 80 * weekly + rng.normal(0, 15, hours)) % 360,
        }
        frame = pd.DataFrame(values)
        frame.insert(0, "time", time)
        frame.insert(0, "ID", station_id)
        random_missing = rng.random((hours, len(FEATURE_COLUMNS))) < 0.10
        frame.loc[:, list(FEATURE_COLUMNS)] = frame.loc[:, list(FEATURE_COLUMNS)].mask(
            random_missing
        )
        # Un apagón contiguo prueba un caso que la interpolación simple maneja mal.
        gap_start = 48 + station_index * 12
        frame.loc[gap_start : gap_start + 11, ["PM10", "PM2.5", "O3"]] = np.nan
        frames.append(frame)

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    LOGGER.info("Dataset sintético creado en %s", path)
    return path


def write_history(path: Path, history: Sequence[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(path, index=False)


def config_from_args(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig(
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        diffusion_steps=args.diffusion_steps,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
    )
    config.validate()
    return config


def add_input_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", type=Path, required=True, help="CSV horario de SIMA.")


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0, help="0 es lo más estable en Windows.")
    parser.add_argument("--seed", type=int, default=42)


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--min-observed", type=float, default=0.2)
    parser.add_argument("--eval-stride", type=int, default=168)


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seq-len", type=int, default=168)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.1)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=24,
        help="24 reduce redundancia; use 1 para replicar el ventaneo del notebook.",
    )


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--mask-ratio", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument(
        "--max-batches", type=int, default=20, help="0 evalúa todo el holdout."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrena, audita e imputa con el modelo híbrido SIMA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Valida y resume el CSV.")
    add_input_argument(inspect_parser)

    train_parser = subparsers.add_parser("train", help="Entrena y guarda un checkpoint.")
    add_input_argument(train_parser)
    add_runtime_arguments(train_parser)
    add_data_arguments(train_parser)
    add_model_arguments(train_parser)
    add_training_arguments(train_parser)
    train_parser.add_argument("--checkpoint", type=Path, default=Path("models/sima.pt"))
    train_parser.add_argument("--output-dir", type=Path, default=Path("outputs/training"))

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Evalúa un checkpoint en un holdout temporal."
    )
    add_input_argument(evaluate_parser)
    add_runtime_arguments(evaluate_parser)
    evaluate_parser.add_argument("--checkpoint", type=Path, default=Path("models/sima.pt"))
    evaluate_parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    evaluate_parser.add_argument("--min-observed", type=float, default=0.2)
    evaluate_parser.add_argument("--eval-stride", type=int, default=168)
    add_evaluation_arguments(evaluate_parser)

    impute_parser = subparsers.add_parser("impute", help="Imputa todo el historial.")
    add_input_argument(impute_parser)
    add_runtime_arguments(impute_parser)
    impute_parser.add_argument("--checkpoint", type=Path, default=Path("models/sima.pt"))
    impute_parser.add_argument(
        "--output", type=Path, default=Path("outputs/sima_dataset_imputado_completo.csv")
    )
    impute_parser.add_argument("--stride", type=int, default=84)
    impute_parser.add_argument("--samples", type=int, default=20)

    run_parser = subparsers.add_parser(
        "run", help="Ejecuta entrenamiento, evaluación e imputación."
    )
    add_input_argument(run_parser)
    add_runtime_arguments(run_parser)
    add_data_arguments(run_parser)
    add_model_arguments(run_parser)
    add_training_arguments(run_parser)
    run_parser.add_argument("--checkpoint", type=Path, default=Path("models/sima.pt"))
    run_parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run_parser.add_argument("--samples", type=int, default=20)
    run_parser.add_argument("--max-batches", type=int, default=20)
    run_parser.add_argument("--impute-stride", type=int, default=84)

    demo_parser = subparsers.add_parser(
        "demo", help="Prueba rápida autocontenida con datos sintéticos."
    )
    demo_parser.add_argument("--output-dir", type=Path, default=Path("outputs/demo"))
    demo_parser.add_argument("--hours", type=int, default=240)
    demo_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    demo_parser.add_argument("--seed", type=int, default=42)
    return parser


def run_training(args: argparse.Namespace) -> tuple[SIMAHybrid, DataBundle, torch.device]:
    if args.epochs < 1 or args.batch_size < 1:
        raise SIMAError("epochs y batch-size deben ser positivos.")
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    config = config_from_args(args)
    bundle = AirQualityDataProcessor(args.input).prepare(
        config.seq_len, args.train_fraction
    )
    LOGGER.info("Dispositivo: %s", device)
    model, history = train_model(
        bundle=bundle,
        config=config,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        mask_ratio=args.mask_ratio,
        train_stride=args.train_stride,
        eval_stride=args.eval_stride,
        min_observed=args.min_observed,
        workers=args.workers,
        seed=args.seed,
    )
    save_checkpoint(
        args.checkpoint, model, bundle.normalizer, history, args.train_fraction, args.seed
    )
    write_history(args.output_dir / "training_history.csv", history)
    return model, bundle, device


def command_inspect(args: argparse.Namespace) -> None:
    stations = AirQualityDataProcessor(args.input).load_raw()
    stations_frame, missing_frame = data_summary(stations)
    print("\nResumen por estación")
    print(stations_frame.to_string(index=False))
    print("\nPorcentaje faltante por variable (promedio entre estaciones)")
    print(
        missing_frame.groupby("feature", sort=False)["missing_percent"]
        .mean()
        .rename("missing_percent")
        .to_string()
    )


def command_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)
    train_fraction = float(payload["training"]["train_fraction"])
    bundle = AirQualityDataProcessor(args.input).prepare(
        model.config.seq_len, train_fraction, normalizer
    )
    summary = evaluate_model(
        model=model,
        bundle=bundle,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        eval_stride=args.eval_stride,
        min_observed=args.min_observed,
        mask_ratio=args.mask_ratio,
        n_samples=args.samples,
        max_batches=args.max_batches,
        workers=args.workers,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def command_impute(args: argparse.Namespace) -> tuple[Path, Path]:
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    model, normalizer, payload = load_checkpoint(args.checkpoint, device)
    train_fraction = float(payload["training"]["train_fraction"])
    bundle = AirQualityDataProcessor(args.input).prepare(
        model.config.seq_len, train_fraction, normalizer
    )
    return impute_dataset(
        model,
        bundle,
        device,
        args.output,
        args.stride,
        args.batch_size,
        args.samples,
    )


def command_run(args: argparse.Namespace) -> None:
    model, bundle, device = run_training(args)
    evaluate_model(
        model=model,
        bundle=bundle,
        device=device,
        output_dir=args.output_dir / "evaluation",
        batch_size=args.batch_size,
        eval_stride=args.eval_stride,
        min_observed=args.min_observed,
        mask_ratio=args.mask_ratio,
        n_samples=args.samples,
        max_batches=args.max_batches,
        workers=args.workers,
        seed=args.seed,
    )
    impute_dataset(
        model,
        bundle,
        device,
        args.output_dir / "sima_dataset_imputado_completo.csv",
        args.impute_stride,
        args.batch_size,
        args.samples,
    )


def command_demo(args: argparse.Namespace) -> None:
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    output_dir: Path = args.output_dir
    input_path = generate_demo_csv(output_dir / "data" / "sima_demo.csv", args.hours, args.seed)
    config = ModelConfig(
        seq_len=24,
        hidden_size=16,
        n_heads=4,
        n_layers=1,
        diffusion_steps=8,
        dropout=0.0,
        residual_scale=0.1,
    )
    bundle = AirQualityDataProcessor(input_path).prepare(config.seq_len, 0.75)
    model, history = train_model(
        bundle,
        config,
        device,
        epochs=1,
        batch_size=16,
        learning_rate=1e-3,
        mask_ratio=0.2,
        train_stride=6,
        eval_stride=24,
        min_observed=0.2,
        workers=0,
        seed=args.seed,
    )
    checkpoint = output_dir / "models" / "sima_demo.pt"
    save_checkpoint(checkpoint, model, bundle.normalizer, history, 0.75, args.seed)
    write_history(output_dir / "training" / "training_history.csv", history)
    summary = evaluate_model(
        model,
        bundle,
        device,
        output_dir / "evaluation",
        batch_size=16,
        eval_stride=24,
        min_observed=0.2,
        mask_ratio=0.2,
        n_samples=4,
        max_batches=3,
        workers=0,
        seed=args.seed,
    )
    imputed, audit = impute_dataset(
        model,
        bundle,
        device,
        output_dir / "sima_demo_imputado.csv",
        stride=12,
        batch_size=16,
        n_samples=4,
    )
    print("\nDemostración completada.")
    print(f"Checkpoint: {checkpoint}")
    print(f"Datos imputados: {imputed}")
    print(f"Auditoría: {audit}")
    print(
        "RMSE de demostración: "
        f"{summary['aggregate_metrics_physical_scale']['rmse']:.4f}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s | %(message)s",
    )
    try:
        if args.command == "inspect":
            command_inspect(args)
        elif args.command == "train":
            run_training(args)
        elif args.command == "evaluate":
            command_evaluate(args)
        elif args.command == "impute":
            command_impute(args)
        elif args.command == "run":
            command_run(args)
        elif args.command == "demo":
            command_demo(args)
        else:  # argparse impide llegar aquí; mantiene exhaustivo el flujo.
            parser.error(f"Comando desconocido: {args.command}")
    except SIMAError as error:
        LOGGER.error("%s", error)
        return 2
    except KeyboardInterrupt:
        LOGGER.error("Ejecución interrumpida por el usuario.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
