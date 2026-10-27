from __future__ import annotations

import numpy as np
import pandas as pd


def metrics(results: pd.DataFrame) -> dict:
    if results.empty:
        raise ValueError("Nessuna previsione da valutare.")
    actual = results["actual_close"].to_numpy(dtype=float)
    previous = results["previous_close"].to_numpy(dtype=float)
    if not np.isfinite(actual).all() or not np.isfinite(previous).all() or (actual <= 0).any() or (previous <= 0).any():
        raise ValueError("I prezzi devono essere finiti e positivi.")

    def evaluate(predicted: np.ndarray) -> dict:
        if not np.isfinite(predicted).all():
            raise ValueError("Le previsioni devono essere finite.")
        error = predicted - actual
        return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
                "mape_percent": float(np.mean(np.abs(error / actual)) * 100),
                "directional_accuracy_percent": float(np.mean(np.sign(predicted - previous) == np.sign(actual - previous)) * 100)}

    model = evaluate(results["predicted_close"].to_numpy(dtype=float))
    return {"count": len(results), "timestral": model}
