## Goal

Add two checkboxes to the GUI to toggle markers on the chart for full moon and new moon dates. If a moon date falls on a weekend or market holiday (not present in the plotted series), use the previous trading day instead.

## UI Changes

* Advanced panel: add two boolean controls

  * “Show Full Moon” (`BooleanVar`: `show_full_moon_var`)

  * “Show New Moon” (`BooleanVar`: `show_new_moon_var`)

* Tooltips: explain behavior and the holiday/weekend fallback.

## Data & Computation

* Determine display date range from the current plot:

  * In `visualize_data`: use the filtered `act_df` (after display filters).

  * In `plot_results`: use `act_ext` (actual data up to forecast end and display filters).

* Compute moon events within `[start_date, end_date]`:

  * Use `ephem` (PyEphem) to iterate:

    * Start from `start_date`: get `ephem.next_full_moon(start)` repeatedly until `> end_date`.

    * Similarly for `ephem.next_new_moon(start)`.

  * Collect all dates as `datetime` (UTC → local naive date OK for plotting daily).

* Trading-day adjustment:

  * Build a sorted `DatetimeIndex` from the actual series (`act_df[time_col]`).

  * For each moon date:

    * If date present in index → keep it.

    * Else, find the previous date in the index (`index[index.get_indexer([date], method='pad')]`).

    * If no previous date exists, skip the marker.

* Intraday handling (optional): if intraday frequency, map to the previous timestamp within that day; otherwise use prior trading day.

## Plot Integration

* Implement `plot_moon_markers(ax, time_series, value_series, full_moons, new_moons)`:

  * For each adjusted moon date, get y-value from `value_series` (e.g., `close`) at that timestamp.

  * Plot with scatter:

    * Full moon: red dots (`color="#d62728"`, size \~30)

    * New moon: purple dots (`color="#9467bd"`, size \~30)

  * Add legend entries only when markers exist.

* Hook calls:

  * `visualize_data`: after plotting actual series, if checkbox(es) enabled, compute events and call `plot_moon_markers`.

  * `plot_results`: after plotting actual/forecast, compute events on `act_ext` and call `plot_moon_markers` when enabled.

## Dependencies

* Add `ephem` to project dependencies (Pipfile `[packages] ephem = "*"`).

* If avoiding new deps: provide a fallback approximate algorithm (e.g., Conway lunar phase) but accuracy will be lower; prefer `ephem` for reliability.

## Performance & Robustness

* Cache computed moon dates per display range to avoid recomputation during small UI updates.

* Handle timezone-naive vs aware datetimes consistently (use naive `datetime` aligned to the `time_col` type).

* Guard against empty data and missing target columns; default to `close`.

## Verification

* Load daily stock data (local or MCP), toggle checkboxes, verify markers align with known full/new moon dates.

* Test weekend/holiday adjustments by selecting ranges with events landing on non-trading days.

* Intraday sample: ensure markers map to a prior intraday timestamp.

## Notes

* Colors and sizes can be adjusted; if you prefer different styles, we can tune after you review.

* If you don’t want an extra dependency, we can implement an internal approximation.

