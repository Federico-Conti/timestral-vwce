# VWCE: backtest walk-forward con Timestral-209M

Esperimento zero-shot, non un sistema di trading. Per ogni seduta Xetra del 2025, il modello riceve **solo** le chiusure reali precedenti, dalla prima osservazione disponibile su Yahoo Finance (nel download eseguito: 29 luglio 2019), e produce una previsione a una seduta. La finestra si espande fino alla seduta precedente al target. Il target non viene passato al modello. La previsione successiva usa la chiusura reale precedente, mai una previsione precedente. Non viene eseguito fine-tuning. Se lo storico supera la lunghezza massima del modello, l'esecuzione si arresta invece di scartare le osservazioni più vecchie.

## Installazione

Servono Python 3.10–3.13, `git`, connessione Internet, spazio per i pesi (~837 MB) e memoria sufficiente. Esempio con Python 3.13:

```sh
python3.13 -m venv .venv
source source.sh
python -m pip install -r requirements.txt
```

Se si usa `uv`: `uv venv --python 3.13 .venv`, quindi `source source.sh` e `uv pip install -r requirements.txt`. `requirements.txt` fissa le dipendenze dirette e il commit del codice Timestral; il download dei pesi usa una revisione Hugging Face specifica. Per riprodurre esattamente anche le dipendenze transitive usare `requirements.lock`. Il caricamento riutilizza i pesi completi già presenti nella cache Hugging Face o riprende un eventuale download parziale. Il file completo viene verificato con SHA-256 prima dell'inferenza. La licenza dei pesi è distinta da quella del codice: verificare i termini nella [model card](https://huggingface.co/mistralai/Timestral-209M).

## Esecuzione

```sh
source source.sh
python -m vwce_forecast --device cpu
```

Opzioni: `--device cuda` per CUDA, `--device mps` per Apple Silicon quando supportato da PyTorch/Timestral; `--seed 42`; `--cache percorso.csv`; `--output cartella`; `--refresh-data` per riscaricare i dati. Non è garantito che CPU, CUDA e MPS producano risultati bit-identici. `python -m vwce_forecast --help` mostra tutte le opzioni.

Il download usa `period="max"`, senza una data iniziale prestabilita, fino al 31 dicembre 2025 incluso e viene conservato in `data/vwce_de_all_history_2025.csv`; Yahoo interpreta l'estremo `2026-01-01` come esclusivo. Il calendario XETR viene confrontato con le date ricevute a partire dalla prima osservazione; le date inattese causano errore. Le sedute senza dati sono conservate come valori mancanti (`NaN`), supportati dall'API ufficiale del modello: il dataset non viene interpolato. Una seduta è esclusa dalla valutazione se manca la sua chiusura oppure quella della seduta precedente; così il rendimento non diventa erroneamente multiseduta. I target esclusi e le ragioni sono riportati in log, metriche e dashboard. In caso di discrepanza va verificato se è errata la fonte o il calendario; nessuno dei due è garantito esente da errori.

Il 24 ottobre 2025 Xetra era aperta, mentre Yahoo non restituisce quella seduta nemmeno con una richiesta limitata a quel giorno. L'endpoint `price_history` di [Deutsche Börse](https://www.boerse-frankfurt.de/) per ISIN `IE00BK5BQT80` e mercato `XETR` restituisce **Open 143,56; High 144,12; Low 143,24; Close 144,10 EUR; Volume 96.556**. La risposta e il relativo URL sono trascritti in `vwce_forecast/supplemental_closes.json`, utilizzato solo quando Yahoo non fornisce il valore. Sei chiusure circostanti corrispondono a Yahoo arrotondate ai centesimi e il codice le verifica prima dell'integrazione. I valori OHLC e volume della fonte ufficiale coincidono con quelli Yahoo nelle sedute limitrofe. La provenienza è indicata in `close_source`, nelle metriche e nel report. L'endpoint non fornisce `Adj Close` o corporate action e questi campi restano vuoti, senza inventarli. L'assenza di un'azione societaria il giorno mancante non è verificabile dai soli dati Yahoo. La pagina di Trading 212 mostra anch'essa 144,10 EUR, ma specifica che i suoi valori sono basati sui prezzi d'acquisto del broker, pertanto non è usata come fonte dei dati. Senza l'integrazione ufficiale si escluderebbero sia il 24 ottobre (target mancante) sia il 27 ottobre (chiusura precedente mancante); con essa entrambe sono valutabili.

Il CSV delle previsioni è salvato atomicamente **dopo ogni seduta**. Per fermarsi dopo una prova e riprendere senza ricalcolare:

```sh
python -m vwce_forecast --max-predictions 2
python -m vwce_forecast
```

Il dataset dei prezzi reali viene esportato prima dell'inferenza; dashboard e report finali vengono generati solo quando tutte le sedute valutabili sono state elaborate. `output_full_history/checkpoint.json` collega il checkpoint all'impronta del dataset e alla configurazione, inclusi politica della finestra, prima data, dimensione del contesto iniziale, modello, revisione, device e seed. I vecchi checkpoint 2024 non sono compatibili e l'output predefinito è separato. Se dati o configurazione cambiano, usare una nuova cartella `--output`: il programma rifiuta di mescolare risultati incompatibili. Conservare anche la cache del dataset per riprodurre l'esecuzione; rinfrescarla può cambiare l'impronta. I log mostrano l'avanzamento. Il tempo di esecuzione dipende dalla macchina; ogni previsione richiede inferenza locale e CPU può essere lenta.

Per scaricare o riprendere separatamente il modello: `python -m vwce_forecast.model_download`. Un'interruzione conserva `~/.cache/vwce-forecast/Timestral-209M-<revisione>/model.safetensors.part`; ripetere il comando per proseguire. Un checksum errato non viene caricato: conservare il file per diagnostica e rimuoverlo o rinominarlo prima di riprovare. Per uno smoke test reale usare `python -m vwce_forecast --max-predictions 1`, poi eseguire il comando senza limite per continuare dallo stesso checkpoint.

## Output

- `output_full_history/vwce_predictions_2025.csv`: previsione mediana, chiusura precedente e reale, errore `previsto - reale`, errore assoluto e percentuale (`errore / reale × 100`), rendimenti percentuali rispetto alla chiusura precedente, direzione corretta e quantili del modello.
- `output_full_history/vwce_actual_2025.csv`: dati reali disponibili, con provenienza e campi mancanti espliciti.
- `output_full_history/metrics_2025.json`: MAE, RMSE, MAPE, accuratezza direzionale di Timestral, storico iniziale, fonti supplementari ed esclusioni.
- `output_full_history/dashboard_2025.html`: dashboard Plotly autonoma e interattiva con prezzi reali e previsioni Timestral. Mostra anche errore giornaliero, errore assoluto e media mobile a 20 sedute, rendimenti previsti e reali.
- `output_full_history/report_2025.md`: metriche sintetiche e limitazioni.

L'accuratezza direzionale è a **tre classi** (salita, discesa, invariato), senza tolleranza. La MAPE usa valori assoluti e non va confusa con `percentage_error`, che è con segno. Rendimenti e percentuali sono espressi in punti percentuali, non frazioni.

## Prezzi, temporalità e limiti

Yahoo viene interrogato con `auto_adjust=False` e `actions=True`. Usiamo `Close` e OHLC originali nella scala restituita da Yahoo e verifichiamo che `Adj Close` coincida con `Close` entro una tolleranza numerica di `1e-5` e che non vi siano dividendi o split nel periodo. Se non è così, il programma si arresta: applicare retroattivamente fattori di rettifica basati su eventi futuri potrebbe introdurre leakage. In quel caso occorre una fonte **point-in-time** o un'implementazione che ricostruisca correttamente le rettifiche disponibili alla data di ciascuna previsione, non una semplice normalizzazione sull'intero campione. Anche in assenza di eventi riportati, Yahoo può correggere prezzi, rettificare retroattivamente i dati e cambiare lo storico: non è una fonte point-in-time certificata. Pertanto la disponibilità storica esatta dei valori e l'assenza di leakage alla fonte non sono dimostrabili da questo dataset.

Il modello è stato pubblicato nell'ottobre 2026, dopo il periodo valutato. La [model card ufficiale](https://huggingface.co/mistralai/Timestral-209M) e il [repository ufficiale](https://github.com/mistralai/timestral) non specificano un cutoff e una composizione dei dati di pretraining sufficienti per escludere la presenza di VWCE o dati del 2025. La possibile contaminazione del pretraining è **ignota**, non una contaminazione accertata: il risultato non è dimostrabilmente out-of-sample rispetto all'addestramento originale. Buona aderenza ai prezzi non implica capacità predittiva dei rendimenti, significatività statistica, profitti o raccomandazioni d'investimento.

Il modello utilizza l'API ufficiale `TimestralForecaster.from_pretrained(percorso_locale)` con i file della revisione Hugging Face fissata e verificata, e `forecast(history, prediction_length=1)`; si salva `forecast.median` e tutti i livelli di `forecast.quantiles`, senza covariate. La libreria applica le proprie opzioni di inferenza predefinite, tra cui symmetric averaging e clipping non-negativo.

## Test

```sh
source source.sh
python -m unittest discover -s tests -v
```

I test usano un modello fittizio, non richiedono il download dei pesi e verificano sequenza temporale, ripresa, identità del checkpoint, metriche, gestione degli invariati e rifiuto dei dati non validi. Non sostituiscono una verifica della fonte né un test out-of-sample del pretraining.
