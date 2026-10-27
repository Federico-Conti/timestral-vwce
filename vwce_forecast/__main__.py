from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .backtest import atomic_write, run_backtest, session_eligibility, verify_checkpoint_identity
from .data import load_data
from .metrics import metrics
from .model import CODE_REVISION, MODEL_ID, MODEL_REVISION, TimestralPredictor
from .report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward zero-shot di Timestral su VWCE.DE")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", type=Path, default=Path("data/vwce_de_all_history_2025.csv"))
    parser.add_argument("--output", type=Path, default=Path("output_full_history"))
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--max-predictions", type=int, help="Limita le nuove previsioni in questa esecuzione, utile per prova e ripresa")
    args = parser.parse_args()
    if args.max_predictions is not None and args.max_predictions < 1:
        parser.error("--max-predictions deve essere positivo")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_data(args.cache, refresh=args.refresh_data)
    args.output.mkdir(parents=True, exist_ok=True)
    initial = data.loc[data.index.year < 2025, "Close"]
    config = {"model_id": MODEL_ID, "model_revision": MODEL_REVISION, "code_revision": CODE_REVISION,
              "device": args.device, "seed": args.seed, "horizon": 1, "price": "yahoo_unadjusted_verified_with_documented_supplement",
              "history_policy": "expanding_from_first_available", "history_first_observation": initial.first_valid_index().strftime("%Y-%m-%d"),
              "initial_observations": int(initial.notna().sum()), "initial_context_steps": len(initial)}
    logging.info("Contesto iniziale dal %s: %d prezzi reali, %d passi totali.", config["history_first_observation"],
                 config["initial_observations"], config["initial_context_steps"])
    verify_checkpoint_identity(data, args.output, config)
    actual = data.loc[(data.index.year == 2025) & data["Close"].notna()].copy()
    atomic_write(args.output / "vwce_actual_2025.csv", actual.to_csv(date_format="%Y-%m-%d", float_format="%.17g"))
    results = run_backtest(data, args.output, config, lambda: TimestralPredictor(args.device, args.seed), args.max_predictions)
    targets, excluded = session_eligibility(data)
    expected = len(targets)
    if len(results) != expected:
        logging.info("Esecuzione parziale: %d/%d; rieseguire lo stesso comando per continuare.", len(results), expected)
        return
    summary = metrics(results)
    summary["excluded_sessions"] = excluded
    summary["calendar_sessions_2025"] = int((data.index.year == 2025).sum())
    summary["history"] = {key: config[key] for key in ("history_policy", "history_first_observation", "initial_observations", "initial_context_steps")}
    supplemental = data.loc[(data.index.year == 2025) & (data["close_source"] == "Deutsche Börse Xetra"), ["Close", "close_source"]]
    summary["supplemental_closes"] = [{"date": date.strftime("%Y-%m-%d"), "close": float(row["Close"]),
                                      "source": row["close_source"]} for date, row in supplemental.iterrows()]
    generate_report(results, summary, args.output)
    logging.info("Backtest completo. Risultati in %s", args.output.resolve())


if __name__ == "__main__":
    main()
