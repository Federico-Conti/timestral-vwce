import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from vwce_forecast.backtest import run_backtest, session_eligibility
from vwce_forecast.data import apply_supplement, validate_data
from vwce_forecast.metrics import metrics
from vwce_forecast.model import TimestralPredictor
from vwce_forecast.report import generate_report


def sample_data():
    index = pd.to_datetime(["2024-12-27", "2024-12-30", "2025-01-02", "2025-01-03", "2025-01-06"])
    closes = [100.0, 101.0, 103.0, 102.0, 104.0]
    return pd.DataFrame({"Open": closes, "High": closes, "Low": closes, "Close": closes,
                         "Adj Close": closes, "Volume": [100] * 5, "Dividends": [0] * 5,
                         "Stock Splits": [0] * 5}, index=index).rename_axis("Date")


class RecordingPredictor:
    def __init__(self):
        self.histories = []

    def predict(self, history):
        self.histories.append(history.copy())
        prediction = float(history[-1] + 1)
        return prediction, {"quantile_0.5": prediction}


class BacktestTests(unittest.TestCase):
    def test_walk_forward_and_resume(self):
        data = sample_data()
        first = RecordingPredictor()
        second = RecordingPredictor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            results = run_backtest(data, path, {"seed": 42}, lambda: first, max_predictions=1)
            self.assertEqual(len(results), 1)
            np.testing.assert_array_equal(first.histories[0], [100, 101])
            results = run_backtest(data, path, {"seed": 42}, lambda: second)
            self.assertEqual(len(results), 3)
            np.testing.assert_array_equal(second.histories[0], [100, 101, 103])
            np.testing.assert_array_equal(second.histories[1], [100, 101, 103, 102])
            self.assertEqual(results["previous_close"].tolist(), [101, 103, 102])
            self.assertEqual(results["actual_close"].tolist(), [103, 102, 104])
            self.assertEqual(results["predicted_close"].tolist(), [102, 104, 103])
            self.assertEqual(results["direction_correct"].tolist(), [True, False, True])
            with self.assertRaisesRegex(ValueError, "incompatibile"):
                run_backtest(data, path, {"seed": 7}, lambda: second)
            changed = data.copy()
            changed.loc[pd.Timestamp("2024-12-30"), "Close"] = 99
            with self.assertRaisesRegex(ValueError, "incompatibile"):
                run_backtest(changed, path, {"seed": 42}, lambda: second)

            summary = metrics(results)
            self.assertEqual(summary["count"], 3)
            self.assertAlmostEqual(summary["timestral"]["mae"], 4 / 3)
            self.assertAlmostEqual(summary["timestral"]["directional_accuracy_percent"], 200 / 3)
            self.assertNotIn("baseline", summary)
            generate_report(results, summary, path)
            self.assertTrue((path / "dashboard_2025.html").is_file())
            self.assertTrue((path / "report_2025.md").is_file())
            self.assertNotIn("Baseline naive", (path / "dashboard_2025.html").read_text())
            self.assertNotIn("naive", (path / "report_2025.md").read_text().lower())

    def test_rejects_adjustments_missing_values_and_duplicates(self):
        frame = sample_data()
        validate_data(frame, check_calendar=False)
        adjusted = frame.copy()
        adjusted.iloc[0, adjusted.columns.get_loc("Adj Close")] *= 0.9
        with self.assertRaisesRegex(ValueError, "rettifiche"):
            validate_data(adjusted, check_calendar=False)
        missing = frame.copy()
        missing.iloc[0, missing.columns.get_loc("Close")] = np.nan
        with self.assertRaisesRegex(ValueError, "mancanti"):
            validate_data(missing, check_calendar=False)
        duplicate = pd.concat([frame, frame.iloc[[-1]]])
        with self.assertRaisesRegex(ValueError, "univoche"):
            validate_data(duplicate, check_calendar=False)

    def test_flat_direction(self):
        results = pd.DataFrame({"actual_close": [100.0], "previous_close": [100.0], "predicted_close": [100.0]})
        summary = metrics(results)
        self.assertEqual(summary["timestral"]["directional_accuracy_percent"], 100.0)
        self.assertEqual(summary["timestral"]["mae"], 0.0)

    def test_missing_session_is_never_imputed_or_used_as_previous_close(self):
        data = sample_data()
        data.loc[pd.Timestamp("2025-01-03"), :] = np.nan
        data.loc[pd.Timestamp("2025-01-07"), :] = [105.0, 105.0, 105.0, 105.0, 105.0, 100, 0, 0]
        predictor = RecordingPredictor()
        targets, excluded = session_eligibility(data)
        self.assertEqual(targets.strftime("%Y-%m-%d").tolist(), ["2025-01-02", "2025-01-07"])
        self.assertEqual([item["date"] for item in excluded], ["2025-01-03", "2025-01-06"])
        with tempfile.TemporaryDirectory() as directory:
            results = run_backtest(data, Path(directory), {}, lambda: predictor)
            self.assertEqual(results["date"].tolist(), ["2025-01-02", "2025-01-07"])
            np.testing.assert_array_equal(predictor.histories[0], [100, 101])
            np.testing.assert_array_equal(predictor.histories[1], [100, 101, 103, np.nan, 104])

    def test_official_forecast_result_shape_and_quantiles(self):
        import torch
        from timestral.forecaster import Forecast

        class StubForecaster:
            max_context = 32768

            def forecast(self, history, prediction_length):
                self.history = history
                self.horizon = prediction_length
                return Forecast(torch.tensor([[[[99.0, 100.0, 101.0]]]]), (0.1, 0.5, 0.9))

        predictor = TimestralPredictor.__new__(TimestralPredictor)
        predictor.forecaster = StubForecaster()
        median, quantiles = predictor.predict(np.array([98.0, 99.0]))
        self.assertEqual(predictor.forecaster.horizon, 1)
        self.assertEqual(median, 100.0)
        self.assertEqual(quantiles, {"quantile_0.1": 99.0, "quantile_0.5": 100.0, "quantile_0.9": 101.0})
        predictor.forecaster.max_context = 1
        with self.assertRaisesRegex(ValueError, "supera il limite"):
            predictor.predict(np.array([98.0, 99.0]))

    def test_earlier_history_is_kept_and_checkpoint_is_incompatible(self):
        recent = sample_data()
        all_history = recent.copy()
        all_history.loc[pd.Timestamp("2019-07-29")] = [71.85, 71.85, 71.85, 71.85, 71.85, 351, 0, 0]
        all_history = all_history.sort_index()
        validate_data(all_history, check_calendar=False)
        predictor = RecordingPredictor()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            run_backtest(recent, path, {"history_policy": "2024"}, lambda: predictor, max_predictions=1)
            with self.assertRaisesRegex(ValueError, "incompatibile"):
                run_backtest(all_history, path, {"history_policy": "expanding_from_first_available"}, lambda: predictor)
            new_path = path / "all"
            run_backtest(all_history, new_path, {"history_policy": "expanding_from_first_available"}, lambda: predictor, max_predictions=1)
            np.testing.assert_array_equal(predictor.histories[-1], [71.85, 100, 101])

    def test_supplement_fills_only_missing_close_with_provenance(self):
        data = sample_data()
        data.loc[pd.Timestamp("2025-01-02"), :] = np.nan
        supplement = {"ticker": "VWCE.DE", "isin": "IE00BK5BQT80", "market": "XETR", "source": "Test",
                      "comparison_yahoo_closes": {"2024-12-30": 101},
                      "rows": {"2025-01-02": {"Open": 102, "High": 104, "Low": 101, "Close": 103, "Volume": 42}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "supplement.json"
            path.write_text(json.dumps(supplement))
            result = apply_supplement(data, path)
            self.assertEqual(result.loc["2025-01-02", "Close"], 103)
            self.assertEqual(result.loc["2025-01-02", "close_source"], "Test")
            self.assertEqual(result.loc["2025-01-02", "Volume"], 42)
            self.assertTrue(pd.isna(result.loc["2025-01-02", "Adj Close"]))
            self.assertEqual(len(session_eligibility(result)[0]), 3)
            supplement["comparison_yahoo_closes"]["2024-12-30"] = 99
            path.write_text(json.dumps(supplement))
            with self.assertRaisesRegex(ValueError, "non è confrontabile"):
                apply_supplement(data, path)


if __name__ == "__main__":
    unittest.main()
