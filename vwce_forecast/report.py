from __future__ import annotations

import html
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .backtest import atomic_write


def generate_report(results, summary: dict, output_dir: Path) -> None:
    atomic_write(output_dir / "metrics_2025.json", json.dumps(summary, indent=2, allow_nan=False) + "\n")
    model = summary["timestral"]
    excluded = summary.get("excluded_sessions", [])
    excluded_text = ", ".join(f"{item['date']} ({item['reason']})" for item in excluded) or "nessuna"
    supplemental = summary.get("supplemental_closes", [])
    supplemental_text = ", ".join(f"{item['date']}: {item['close']:.2f} EUR ({item['source']})" for item in supplemental) or "nessuna"
    history = summary.get("history", {})
    report = f"""# Report VWCE — walk-forward 2025

- Previsioni: {summary['count']}
- Contesto iniziale: {history.get('initial_observations', 'n.d.')} osservazioni dalla prima data {history.get('history_first_observation', 'n.d.')}.
- Sedute 2025 escluse: {excluded_text}.
- Chiusure da fonte supplementare: {supplemental_text}.
- MAE (EUR): Timestral {model['mae']:.4f}.
- RMSE (EUR): Timestral {model['rmse']:.4f}.
- MAPE: Timestral {model['mape_percent']:.4f}%.
- Accuratezza direzionale (tre classi: salita/discesa/invariato): Timestral {model['directional_accuracy_percent']:.2f}%.

Le metriche confrontano le previsioni di Timestral con i prezzi reali. I prezzi contigui possono essere simili anche senza capacità di previsione dei rendimenti; verificare anche gli altri indicatori e grafici. Non si valutano costi di transazione né strategie di investimento.

Il modello è successivo al 2025 e non sono pubblicati nel model card dettagli sufficienti a escludere che abbia incontrato questi dati in pretraining. Questo non è quindi un test dimostrabilmente out-of-sample rispetto al pretraining. Yahoo Finance può correggere retroattivamente i dati: non si tratta di un archivio point-in-time certificato. Consultare il README per le altre limitazioni.
"""
    atomic_write(output_dir / "report_2025.md", report)

    figure = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.045,
                           subplot_titles=("Prezzi: reali e Timestral", "Errore giornaliero (previsto − reale)",
                                           "Errore assoluto e media mobile a 20 sedute", "Rendimenti giornalieri (%)"))
    dates = results["date"]
    figure.add_trace(go.Scatter(x=dates, y=results["actual_close"], name="Prezzo reale", line={"color": "#146c94"}), row=1, col=1)
    figure.add_trace(go.Scatter(x=dates, y=results["predicted_close"], name="Timestral", line={"color": "#ee8f20"}), row=1, col=1)
    figure.add_trace(go.Bar(x=dates, y=results["error"], name="Errore", marker_color="#bc5466"), row=2, col=1)
    figure.add_hline(y=0, line_dash="dash", line_color="#444", row=2, col=1)
    figure.add_trace(go.Scatter(x=dates, y=results["absolute_error"], name="Errore assoluto", line={"color": "#888"}), row=3, col=1)
    figure.add_trace(go.Scatter(x=dates, y=results["absolute_error"].rolling(20, min_periods=1).mean(), name="Media mobile 20 sedute", line={"color": "#b34e22"}), row=3, col=1)
    figure.add_trace(go.Scatter(x=dates, y=results["actual_return"], name="Rendimento reale", line={"color": "#146c94"}), row=4, col=1)
    figure.add_trace(go.Scatter(x=dates, y=results["predicted_return"], name="Rendimento previsto", line={"color": "#ee8f20"}), row=4, col=1)
    for row, label in enumerate(["EUR", "EUR", "EUR", "%"], start=1):
        figure.update_yaxes(title_text=label, row=row, col=1)
    figure.update_layout(height=1350, template="plotly_white", hovermode="x unified", legend={"orientation": "h"}, margin={"t": 70})
    table_rows = []
    labels = [("mae", "MAE (EUR)"), ("rmse", "RMSE (EUR)"), ("mape_percent", "MAPE (%)"),
              ("directional_accuracy_percent", "Accuratezza direzionale (%)")]
    for key, label in labels:
        table_rows.append(f"<tr><th>{html.escape(label)}</th><td>{model[key]:.4f}</td></tr>")
    table = "".join(table_rows)
    chart = figure.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True})
    page = f"""<!doctype html><html lang="it"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>VWCE — Backtest 2025</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:24px;color:#172b36}} table{{border-collapse:collapse;width:100%;max-width:850px}} th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:right}} th:first-child{{text-align:left}} .note{{color:#52636b;font-size:.9rem}}</style></head>
<body><h1>VWCE — Backtest 2025</h1><p>{summary['count']} previsioni confrontate con i prezzi reali.</p>
<table><thead><tr><th>Metrica</th><th>Timestral</th></tr></thead><tbody>{table}</tbody></table>
<p class="note">Sedute escluse: {html.escape(excluded_text)}. Chiusure supplementari: {html.escape(supplemental_text)}; per esse Adj Close e corporate action non sono disponibili. Direzione: salita, discesa o invariato. Il test non è dimostrabilmente out-of-sample rispetto al pretraining. Non costituisce consulenza finanziaria.</p>{chart}</body></html>"""
    atomic_write(output_dir / "dashboard_2025.html", page)
