## Goal
Update `examples/stock_forecast_gui_Kronos.py` to perform forecasts using the local Kronos model/tokenizer and predictor, following `examples/prediction_example.py` as the reference.

## Imports & Initialization
- Add project-root path injection (based on `__file__`) so `from model import ...` works regardless of working directory.
- Import `torch` and `from model import Kronos, KronosTokenizer, KronosPredictor`.
- On app startup (in `App.__init__`), lazily load Kronos components:
  - `device = "cuda:0" if torch.cuda.is_available() else "cpu"`
  - `self.kronos_tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")`
  - `self.kronos_model = Kronos.from_pretrained("NeoQuasar/Kronos-small")`
  - `self.kronos_predictor = KronosPredictor(self.kronos_model, self.kronos_tokenizer, device=device, max_context=512)`

## Data Requirements & Preparation
- Kronos expects multivariate OHLCV+amount with timestamps:
  - Required price columns: `open, high, low, close`; optional: `volume, amount`.
  - If `volume` or `amount` missing, fill per `KronosPredictor.predict` behavior (volume/amount zeros or `amount = volume * mean(price_cols)`).
- Use current GUI-selected series and range, then:
  - Determine `freq`: use existing `_suggest_freq()` result or `self.freq_var` input; fallback to `'B'` for daily.
  - Compute `lookback_len`: default `400`, but clamp to available history (`min(400, len(df))`).
  - Build `x_df = df.iloc[:lookback_len][['open','high','low','close','volume','amount']]` (create missing columns as needed).
  - Build `x_timestamp = df.iloc[:lookback_len][time_col]`.
  - Build `y_timestamp` for future horizon `h = int(self.h_var.get())`:
    - If the GUI selection provides future rows, use them; else generate with `pd.date_range(start=last_time + one_step, periods=h, freq=freq)`.

## Forecast Execution (replace Nixtla)
- Replace `run_forecast` internals that call `NixtlaClient.forecast`:
  - After existing cleaning/imputation, call:
    - `pred_df = self.kronos_predictor.predict(df=x_df, x_timestamp=x_timestamp, y_timestamp=y_timestamp, pred_len=h, T=1.0, top_p=0.9, sample_count=1, verbose=True)`
  - Store `self.last_fcst_df = pred_df`, and set tracking variables (`id_col`, `time_col`, `target_col`), with `target_col='close'`.

## Plotting Updates
- Update `plot_results` to handle Kronos output:
  - Plot actual `target_col` (e.g., `close`) and predicted `close` from `pred_df`.
  - Remove interval/quantile shading logic (specific to TimeGPT), or guard it behind conditionals when those columns are absent.
  - Title and labels adjusted to “Kronos Forecast”.

## UI Impact (Minimal)
- Keep existing UI fields; do not add new widgets. Use `h` as horizon and `freq` for future timestamp generation.
- If required columns are missing in user data, synthesize minimally:
  - `open/high/low = close` when absent; `volume = 0`; `amount = close * volume`.
- Provide user feedback (status bar) when columns are synthesized.

## Error Handling
- Validate data length and columns; show `messagebox.showerror` on:
  - Empty DataFrame after filtering
  - Non-datetime `time_col` or missing `close`
  - Insufficient history (< 10 rows), suggest loading more or reducing `h`

## Verification
- Run locally with synthetic or sample CSV/TSV:
  - Load data via GUI, set `h` (e.g., 24), run forecast; observe plotted actual vs predicted close.
- Also verify launching from different working dirs (import robustness via `sys.path` fix).

## Notes
- Keeps current advanced options but ignores Nixtla-specific features when using Kronos.
- Device auto-selects CPU when CUDA is unavailable.
- Follows the sampling parameters from `prediction_example.py` (`T=1.0`, `top_p=0.9`, `sample_count=1`).