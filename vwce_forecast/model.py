from __future__ import annotations

import random

import numpy as np

MODEL_ID = "mistralai/Timestral-209M"
MODEL_REVISION = "9874d252ab0aae16f7f7c999f98f98d6ad7d8365"
CODE_REVISION = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"


class TimestralPredictor:
    def __init__(self, device: str, seed: int):
        import torch
        from timestral import TimestralForecaster
        from .model_download import ensure_model_files

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA non disponibile.")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS non disponibile.")
        self.forecaster = TimestralForecaster.from_pretrained(ensure_model_files(), device=device)

    def predict(self, history: np.ndarray) -> tuple[float, dict[str, float]]:
        if len(history) > self.forecaster.max_context:
            raise ValueError(f"Lo storico ({len(history)} sedute) supera il limite del modello ({self.forecaster.max_context}); nessun dato viene scartato automaticamente.")
        forecast = self.forecaster.forecast(np.asarray(history, dtype=np.float32), prediction_length=1)
        median = float(forecast.median[0, 0, 0].item())
        quantiles = {f"quantile_{level:g}": float(forecast.quantiles[0, 0, 0, index].item())
                     for index, level in enumerate(forecast.quantile_levels)}
        if not np.isfinite(median) or not all(np.isfinite(value) for value in quantiles.values()):
            raise ValueError("Il modello ha restituito una previsione non finita.")
        return median, quantiles
