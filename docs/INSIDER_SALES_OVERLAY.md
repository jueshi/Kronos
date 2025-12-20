# Insider Sales Overlay – Fixes and Usage

## Overview
- Adds a “Show Insider Sales” checkbox to overlay insider sale transactions on the chart.
- Fetches sales via `yfinance` with a Finviz HTML fallback.
- Maps sale dates to the nearest plotted trading timestamp and renders bubbles sized by transaction value, with optional vertical lines.

## Fixes Implemented
- Reactive checkbox callbacks:
  - `ttk.Checkbutton(..., command=self._on_overlay_toggle)` triggers immediate re-render when toggled.
  - Code: `examples/stock_forecast_gui_Kronos.py:268–271`
- Robust data fetching and fallback:
  - Primary: `yfinance.Ticker(symbol).insider_transactions`
  - Fallback: Finviz “Insider Trading” table → filter “Sale”
  - Code: `examples/stock_forecast_gui_Kronos.py:1386` and `examples/insider_sale_data.py:124–221`
- Index-based alignment for plotting:
  - Uses a sorted `DatetimeIndex` from the plotted series and pads each sale date to the previous timestamp by index position.
  - Avoids exact timestamp equality issues and ensures a y-value is found when possible.
  - Code: `examples/stock_forecast_gui_Kronos.py:1482–1519`
- Status and debug output:
  - Status bar message always updated with fetched/mapped/plotted counts.
  - Console prints detailed “Debug:” logs for toggle, fetch, and plot steps.
  - Code: `examples/stock_forecast_gui_Kronos.py:1521–1544`, `1482–1519`
- Fallback symbol detection:
  - If the Symbol field is empty, uses `unique_id` from the data (e.g., MCP-loaded series) as a fallback ticker.
  - Code: `examples/stock_forecast_gui_Kronos.py:1488–1498`

## Usage
- Enable “Show Insider Sales” in Advanced.
- Set `Symbol` to the ticker (e.g., `AAPL`). If empty, ensure your data includes `unique_id` with the ticker.
- Optional: enable “VLines” to draw vertical lines at sale dates.
- Click “Visualize Data” or “Run Forecast”. The overlay refreshes automatically on checkbox toggle.

## Debug Output
- Toggle: prints current overlay states.
- Fetch: prints symbol and fetched row count.
- Plot: prints scatter and vline counts.
- Status: “Insider: fetched TOTAL total, mapped M in range, plotted K markers (scatter S)”.
- Code: `examples/stock_forecast_gui_Kronos.py:1521–1544`, `1482–1519`

## Troubleshooting
- No bubbles:
  - Ensure the target column has values at mapped timestamps (`Target Col`).
  - Verify the display range includes sale dates; adjust `Display From/To`.
  - Check terminal logs for “Debug:” lines and the status bar message.
- No data fetched:
  - yfinance may rate-limit; try again or use Finviz fallback automatically.
  - Confirm `Symbol` matches the plotted series (e.g., `ALAB`).

## Code References
- Checkbox wiring: `examples/stock_forecast_gui_Kronos.py:268–271`
- Earnings overlay (for parity): `examples/stock_forecast_gui_Kronos.py:1280–1361`
- Insider fetch: `examples/stock_forecast_gui_Kronos.py:1386`
- Insider plot: `examples/stock_forecast_gui_Kronos.py:1482–1519`
- Overlay toggle: `examples/stock_forecast_gui_Kronos.py:1521–1544`
