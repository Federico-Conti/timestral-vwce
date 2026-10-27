from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
BASE_COLUMNS = ["date", "previous_close", "predicted_close", "actual_close", "error", "absolute_error",
                "percentage_error", "predicted_return", "actual_return", "direction_correct"]


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dataset_fingerprint(data: pd.DataFrame) -> str:
    canonical = data.to_csv(date_format="%Y-%m-%d", float_format="%.17g")
    return hashlib.sha256(canonical.encode()).hexdigest()


def session_eligibility(data: pd.DataFrame) -> tuple[pd.DatetimeIndex, list[dict]]:
    candidates = data.index[data.index.year == 2025]
    eligible = []
    excluded = []
    for target in candidates:
        position = data.index.get_loc(target)
        if not np.isfinite(data["Close"].iloc[position]):
            excluded.append({"date": target.strftime("%Y-%m-%d"), "reason": "chiusura target mancante"})
        elif position == 0 or not np.isfinite(data["Close"].iloc[position - 1]):
            excluded.append({"date": target.strftime("%Y-%m-%d"), "reason": "chiusura della seduta precedente mancante"})
        else:
            eligible.append(target)
    return pd.DatetimeIndex(eligible, name=data.index.name), excluded


def verify_checkpoint_identity(data: pd.DataFrame, output_dir: Path, config: dict) -> dict:
    identity = {"schema": 1, "data_sha256": dataset_fingerprint(data), **config}
    path = output_dir / "vwce_predictions_2025.csv"
    metadata_path = output_dir / "checkpoint.json"
    if path.exists() and not metadata_path.exists():
        raise ValueError("Metadati del checkpoint assenti: usare un'altra cartella di output.")
    if metadata_path.exists() and json.loads(metadata_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Checkpoint incompatibile con dataset o configurazione. Usare un'altra cartella di output.")
    return identity


def run_backtest(data: pd.DataFrame, output_dir: Path, config: dict,
                 predictor_factory: Callable, max_predictions: int | None = None) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "vwce_predictions_2025.csv"
    metadata_path = output_dir / "checkpoint.json"
    identity = verify_checkpoint_identity(data, output_dir, config)
    targets, excluded = session_eligibility(data)
    if excluded:
        LOGGER.warning("Escluse %d sedute non valutabili: %s", len(excluded), excluded)
    if not len(targets) or not (data.index.year == 2024).any():
        raise ValueError("Il backtest richiede dati del 2024 e del 2025.")
    if metadata_path.exists() and not path.exists():
        atomic_write(path, pd.DataFrame(columns=BASE_COLUMNS).to_csv(index=False))
    if metadata_path.exists():
        existing = pd.read_csv(path, float_precision="round_trip")
        if not set(BASE_COLUMNS).issubset(existing.columns):
            raise ValueError("Checkpoint senza tutte le colonne richieste.")
        dates = pd.to_datetime(existing["date"], errors="raise")
        if len(existing) > len(targets) or not pd.DatetimeIndex(dates).equals(targets[:len(existing)]):
            raise ValueError("Le date del checkpoint non sono un prefisso valido delle sedute.")
        actual = data.loc[targets[:len(existing)], "Close"].to_numpy()
        positions = data.index.get_indexer(targets[:len(existing)])
        previous = data["Close"].iloc[positions - 1].to_numpy()
        if not np.allclose(existing["actual_close"], actual) or not np.allclose(existing["previous_close"], previous):
            raise ValueError("I prezzi nel checkpoint non corrispondono ai dati.")
        if len(existing) and (not np.isfinite(existing.select_dtypes(include=["number"]).to_numpy()).all()):
            raise ValueError("Il checkpoint contiene valori non finiti.")
        rows = existing.to_dict("records")
    else:
        atomic_write(metadata_path, json.dumps(identity, indent=2, sort_keys=True) + "\n")
        atomic_write(path, pd.DataFrame(columns=BASE_COLUMNS).to_csv(index=False))
        rows = []
    remaining = len(targets) - len(rows)
    limit = remaining if max_predictions is None else min(remaining, max_predictions)
    if limit <= 0:
        return pd.DataFrame(rows, columns=BASE_COLUMNS if not rows else None)
    predictor = predictor_factory()
    for target in targets[len(rows):len(rows) + limit]:
        position = data.index.get_loc(target)
        history = data["Close"].iloc[:position].to_numpy(dtype=float)
        if not len(history) or not np.isfinite(history[-1]) or not (data.index[position - 1] < target):
            raise ValueError("Storico temporale non valido.")
        predicted, quantiles = predictor.predict(history.copy())
        actual = float(data["Close"].iloc[position])
        previous = float(history[-1])
        if not np.isfinite(predicted) or not all(np.isfinite(value) for value in quantiles.values()):
            raise ValueError("Previsione non finita.")
        error = predicted - actual
        predicted_return = (predicted / previous - 1) * 100
        actual_return = (actual / previous - 1) * 100
        row = {"date": target.strftime("%Y-%m-%d"), "previous_close": previous, "predicted_close": predicted,
               "actual_close": actual, "error": error, "absolute_error": abs(error),
               "percentage_error": error / actual * 100, "predicted_return": predicted_return,
               "actual_return": actual_return, "direction_correct": bool(np.sign(predicted - previous) == np.sign(actual - previous)),
               **quantiles}
        rows.append(row)
        atomic_write(path, pd.DataFrame(rows).to_csv(index=False, float_format="%.17g"))
        LOGGER.info("%d/%d %s: previsione %.4f, reale %.4f", len(rows), len(targets), target.date(), predicted, actual)
    return pd.DataFrame(rows)
