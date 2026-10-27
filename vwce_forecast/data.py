from __future__ import annotations

import json
import logging
from pathlib import Path

import exchange_calendars as calendars
import numpy as np
import pandas as pd
import yfinance as yf

LOGGER = logging.getLogger(__name__)
END_EXCLUSIVE = "2026-01-01"
COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits"]
SUPPLEMENT_FILE = Path(__file__).with_name("supplemental_closes.json")


def validate_data(frame: pd.DataFrame, *, check_calendar: bool = True) -> pd.DataFrame:
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Colonne mancanti: {sorted(missing)}")
    data = frame.loc[:, COLUMNS].copy()
    data.index = pd.to_datetime(data.index, errors="raise")
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data.index = data.index.normalize()
    data.index.name = "Date"
    if data.index.has_duplicates or not data.index.is_monotonic_increasing:
        raise ValueError("Le date devono essere univoche e in ordine crescente.")
    if data.empty or data.index.max() >= pd.Timestamp(END_EXCLUSIVE):
        raise ValueError("Date vuote o successive al 31 dicembre 2025.")
    for column in COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="raise")
    unavailable = data.isna().all(axis=1)
    observed = data.loc[~unavailable]
    if observed.empty:
        raise ValueError("Il dataset non contiene osservazioni.")
    data = data.loc[observed.index.min():]
    if not np.isfinite(observed.to_numpy(dtype=float)).all():
        raise ValueError("Il dataset contiene valori mancanti o non finiti: non viene effettuata interpolazione.")
    if (observed[["Open", "High", "Low", "Close", "Adj Close"]] <= 0).any().any() or (observed["Volume"] < 0).any():
        raise ValueError("Prezzi non positivi o volume negativo.")
    if not (observed.index.year == 2024).any() or not (observed.index.year == 2025).any():
        raise ValueError("Sono necessarie osservazioni sia nel 2024 sia nel 2025.")

    ratios = observed["Adj Close"] / observed["Close"]
    if not np.allclose(ratios, 1.0, atol=1e-5, rtol=1e-5) or (observed[["Dividends", "Stock Splits"]] != 0).any().any():
        raise ValueError("Rilevate rettifiche o corporate action: servono dati point-in-time; il backtest si arresta per evitare leakage.")

    if check_calendar:
        sessions = calendars.get_calendar("XETR").sessions_in_range(observed.index.min(), "2025-12-31")
        sessions = pd.DatetimeIndex(sessions).tz_localize(None)
        unexpected = data.index.difference(sessions)
        if len(unexpected):
            raise ValueError(f"Date inattese rispetto al calendario XETR: {unexpected.strftime('%Y-%m-%d').tolist()}.")
        data = data.reindex(sessions)
        data.index.name = "Date"
        absent = data.index[data["Close"].isna()]
        if len(absent):
            LOGGER.warning("Sedute XETR senza dati Yahoo: %s. Nessuna interpolazione; saranno escluse se non recuperabili.",
                           absent.strftime("%Y-%m-%d").tolist())
    return data


def apply_supplement(data: pd.DataFrame, supplement_path: Path = SUPPLEMENT_FILE) -> pd.DataFrame:
    data = data.copy()
    data["close_source"] = np.where(data["Close"].notna(), "Yahoo Finance", "non disponibile")
    if not supplement_path.exists():
        return data
    supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
    if (supplement["ticker"], supplement["isin"], supplement["market"]) != ("VWCE.DE", "IE00BK5BQT80", "XETR"):
        raise ValueError("La fonte supplementare non corrisponde a VWCE.DE su Xetra.")
    for date, close in supplement["comparison_yahoo_closes"].items():
        date = pd.Timestamp(date)
        if date not in data.index or not np.isfinite(data.at[date, "Close"]) or abs(data.at[date, "Close"] - close) > 0.011:
            raise ValueError(f"La fonte supplementare non è confrontabile con Yahoo alla data {date.date()}.")
    for date, values in supplement["rows"].items():
        date = pd.Timestamp(date)
        if (date not in data.index or set(values) != {"Open", "High", "Low", "Close", "Volume"}
                or not all(np.isfinite(value) for value in values.values())
                or any(values[column] <= 0 for column in ("Open", "High", "Low", "Close")) or values["Volume"] < 0):
            raise ValueError(f"Chiusura supplementare non valida per {date.date()}.")
        if pd.isna(data.at[date, "Close"]):
            for column, value in values.items():
                data.at[date, column] = float(value)
            data.at[date, "close_source"] = supplement["source"]
            LOGGER.warning("Seduta %s recuperata da %s: chiusura %.2f EUR; rettifiche e corporate action restano mancanti.",
                           date.date(), supplement["source"], values["Close"])
    return data


def load_data(cache_path: Path, refresh: bool = False) -> pd.DataFrame:
    if cache_path.exists() and not refresh:
        LOGGER.info("Carico la cache %s", cache_path)
        raw = pd.read_csv(cache_path, index_col="Date", parse_dates=["Date"], float_precision="round_trip")
    else:
        LOGGER.info("Scarico VWCE.DE da Yahoo Finance")
        raw = yf.download("VWCE.DE", period="max", end=END_EXCLUSIVE, interval="1d", auto_adjust=False,
                          actions=True, progress=False, threads=False, multi_level_index=False)
        if raw.empty:
            raise RuntimeError("Yahoo Finance non ha restituito dati per VWCE.DE.")
    data = validate_data(raw)
    if refresh or not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(cache_path, float_format="%.17g")
    return apply_supplement(data)
