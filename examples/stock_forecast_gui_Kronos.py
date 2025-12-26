import os
import json
import sys
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
from model import Kronos, KronosTokenizer, KronosPredictor
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from nixtla import NixtlaClient
class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip = None
        self.id = None
        widget.bind("<Enter>", self.enter)
        widget.bind("<Leave>", self.leave)
    def enter(self, event=None):
        self.schedule()
    def leave(self, event=None):
        self.unschedule()
        self.hide()
    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.show)
    def unschedule(self):
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
    def show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = ttk.Label(tw, text=self.text, background="#ffffe0", relief=tk.SOLID, borderwidth=1)
        label.pack(ipadx=6, ipady=4)
    def hide(self):
        if self.tip:
            self.tip.destroy()
            self.tip = None

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Kronos Stock Forecast GUI")
        self.df = None
        self.filtered_df = None
        self._tooltips = []
        self.api_key_var = tk.StringVar(value=self._load_api_key())
        self.base_url_var = tk.StringVar(value=os.getenv("NIXTLA_BASE_URL", ""))
        self.symbol_var = tk.StringVar(value="QQQ")
        self.file_path_var = tk.StringVar()
        self.id_col_var = tk.StringVar(value="unique_id")
        self.time_col_var = tk.StringVar(value="ds")
        self.target_col_var = tk.StringVar(value="y")
        self.series_id_var = tk.StringVar()
        self.stock_period_var = tk.StringVar(value="1y")
        self.freq_var = tk.StringVar(value="B")
        self.h_var = tk.IntVar(value=30)
        self.model_var = tk.StringVar(value="timegpt-1")
        self.level_var = tk.StringVar(value="80,90")
        self.quantiles_var = tk.StringVar(value="")
        self.start_var = tk.StringVar(value="")
        self.end_var = tk.StringVar(value="")
        self.add_history_var = tk.BooleanVar(value=False)
        self.date_features_var = tk.BooleanVar(value=False)
        self.date_features_oh_var = tk.BooleanVar(value=False)
        self.clean_ex_first_var = tk.BooleanVar(value=True)
        self.feature_contrib_var = tk.BooleanVar(value=False)
        self.multivariate_var = tk.BooleanVar(value=False)
        self.auto_fix_ts_var = tk.BooleanVar(value=True)
        self.impute_target_var = tk.BooleanVar(value=True)
        self.impute_method_var = tk.StringVar(value="ffill_bfill")
        self.auto_fix_ts_var = tk.BooleanVar(value=True)
        self.finetune_steps_var = tk.IntVar(value=0)
        self.finetune_depth_var = tk.IntVar(value=1)
        self.finetune_loss_var = tk.StringVar(value="default")
        self.finetuned_model_id_var = tk.StringVar(value="")
        self.model_params_var = tk.StringVar(value="")
        self.advanced_visible = True
        self.figure = None
        self.canvas = None
        self.display_start_var = tk.StringVar(value="")
        self.display_end_var = tk.StringVar(value="")
        self.last_fcst_df = None
        self.last_id_col = None
        self.last_time_col = None
        self.last_target_col = None
        self.last_levels = None
        self.last_quantiles = None
        self.kronos_tokenizer_id_var = tk.StringVar(value="NeoQuasar/Kronos-Tokenizer-base")
        self.kronos_model_id_var = tk.StringVar(value="NeoQuasar/Kronos-small")
        self.kronos_device_var = tk.StringVar(value=("cuda:0" if torch.cuda.is_available() else "cpu"))
        self.max_context_var = tk.IntVar(value=512)
        self.clip_var = tk.IntVar(value=5)
        self.temp_var = tk.DoubleVar(value=0.8)
        self.top_p_var = tk.DoubleVar(value=0.9)
        self.top_k_var = tk.IntVar(value=0)
        self.sample_count_var = tk.IntVar(value=3)
        self.show_full_moon_var = tk.BooleanVar(value=True)
        self.show_new_moon_var = tk.BooleanVar(value=True)
        self.show_first_quarter_moon_var = tk.BooleanVar(value=True)
        self.show_last_quarter_moon_var = tk.BooleanVar(value=True)
        self.marker_mode_var = tk.BooleanVar(value=False)
        self.marker_text_var = tk.StringVar(value="")
        self.marker_series_var = tk.StringVar(value="Actual")
        self.show_marker_vlines_var = tk.BooleanVar(value=False)
        self.show_earnings_var = tk.BooleanVar(value=False)
        self.show_insider_var = tk.BooleanVar(value=False)
        self.show_insider_proposed_var = tk.BooleanVar(value=False)
        self.insider_tooltips_var = tk.BooleanVar(value=False)
        self.show_volume_dot_var = tk.BooleanVar(value=False)
        self.show_insider_proposed_var = tk.BooleanVar(value=False)
        self.show_pe_eps_var = tk.BooleanVar(value=False)
        self._build_ui()

        # Kronos initialization
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._kronos_device = device
            self.kronos_tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            self.kronos_model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            self.kronos_predictor = KronosPredictor(self.kronos_model, self.kronos_tokenizer, device=device, max_context=512)
        except Exception:
            self.kronos_tokenizer = None
            self.kronos_model = None
            self.kronos_predictor = None
        self.table_win = None
        self.last_act_df = None
        self.last_act_time_col = None
        self.last_act_target = None
        self.markers = []
        self._marker_pending_text = None
        self._marker_pending_series = None
        self._marker_click_cid = None

    def _build_ui(self):
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=8, pady=8)
        self.load_btn = ttk.Button(top, text="Load Local Data", command=self.load_file)
        self.load_btn.grid(row=0, column=0, padx=6, sticky=tk.W)
        ttk.Label(top, text="Symbol").grid(row=0, column=1, sticky=tk.W)
        self.symbol_entry = ttk.Entry(top, textvariable=self.symbol_var, width=10)
        self.symbol_entry.grid(row=0, column=2, sticky=tk.W)
        ttk.Label(top, text="Length").grid(row=0, column=3, sticky=tk.W)
        self.stock_period_cb = ttk.Combobox(top, textvariable=self.stock_period_var, values=["1d","5d","1mo","3mo","6mo","1y","2y","5y","max"], width=8)
        self.stock_period_cb.grid(row=0, column=4, sticky=tk.W)
        self.load_stock_btn = ttk.Button(top, text="Download (MCP)", command=self.load_stock_data)
        self.load_stock_btn.grid(row=0, column=5, padx=6, sticky=tk.W)

        cols = ttk.Frame(self.root)
        cols.pack(fill=tk.X, padx=8)
        ttk.Label(cols, text="Time Col").grid(row=0, column=0, sticky=tk.W)
        self.time_col_cb = ttk.Combobox(cols, textvariable=self.time_col_var, values=[], width=20)
        self.time_col_cb.grid(row=0, column=1)
        ttk.Label(cols, text="Target Col").grid(row=0, column=2, sticky=tk.W)
        self.target_col_cb = ttk.Combobox(cols, textvariable=self.target_col_var, values=[], width=20)
        self.target_col_cb.grid(row=0, column=3)
        ttk.Label(cols, text="ID Col").grid(row=0, column=4, sticky=tk.W)
        self.id_col_cb = ttk.Combobox(cols, textvariable=self.id_col_var, values=[], width=20)
        self.id_col_cb.grid(row=0, column=5)
        ttk.Label(cols, text="Series ID").grid(row=0, column=6, sticky=tk.W)
        self.series_id_cb = ttk.Combobox(cols, textvariable=self.series_id_var, values=[], width=20)
        self.series_id_cb.grid(row=0, column=7)

        rng = ttk.Frame(self.root)
        rng.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(rng, text="Train From").grid(row=0, column=0)
        self.start_entry = ttk.Entry(rng, textvariable=self.start_var, width=20)
        self.start_entry.grid(row=0, column=1)
        ttk.Label(rng, text="Train To").grid(row=0, column=2)
        self.end_entry = ttk.Entry(rng, textvariable=self.end_var, width=20)
        self.end_entry.grid(row=0, column=3)
        ttk.Label(rng, text="Freq").grid(row=0, column=4)
        self.freq_entry = ttk.Entry(rng, textvariable=self.freq_var, width=10)
        self.freq_entry.grid(row=0, column=5)
        ttk.Button(rng, text="Use Full Range", command=self.use_full_range).grid(row=0, column=6, padx=6)

        pred = ttk.Frame(self.root)
        pred.pack(fill=tk.X, padx=8)
        ttk.Label(pred, text="Horizon").grid(row=0, column=0)
        self.h_entry = ttk.Entry(pred, textvariable=self.h_var, width=8)
        self.h_entry.grid(row=0, column=1)
        # Kronos does not use external model/intervals; keep Horizon only

        kconf = ttk.Frame(self.root)
        kconf.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(kconf, text="Tokenizer").grid(row=0, column=0, sticky=tk.W)
        self.k_tokenizer_cb = ttk.Combobox(kconf, textvariable=self.kronos_tokenizer_id_var, values=[
            "NeoQuasar/Kronos-Tokenizer-base",
            "NeoQuasar/Kronos-Tokenizer-2k",
        ], width=32)
        self.k_tokenizer_cb.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(kconf, text="Model").grid(row=0, column=2, sticky=tk.W)
        self.k_model_cb = ttk.Combobox(kconf, textvariable=self.kronos_model_id_var, values=[
            "NeoQuasar/Kronos-small",
            "NeoQuasar/Kronos-base",
            "NeoQuasar/Kronos-mini",
        ], width=28)
        self.k_model_cb.grid(row=0, column=3, sticky=tk.W)
        ttk.Label(kconf, text="Device").grid(row=0, column=4, sticky=tk.W)
        self.k_device_cb = ttk.Combobox(kconf, textvariable=self.kronos_device_var, values=["cpu","cuda:0"], width=8)
        self.k_device_cb.grid(row=0, column=5, sticky=tk.W)
        ttk.Label(kconf, text="MaxCtx").grid(row=1, column=0, sticky=tk.W)
        self.k_maxctx_entry = ttk.Entry(kconf, textvariable=self.max_context_var, width=8)
        self.k_maxctx_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(kconf, text="Clip").grid(row=1, column=2, sticky=tk.W)
        self.k_clip_entry = ttk.Entry(kconf, textvariable=self.clip_var, width=8)
        self.k_clip_entry.grid(row=1, column=3, sticky=tk.W)
        ttk.Label(kconf, text="Temp").grid(row=1, column=4, sticky=tk.W)
        self.k_temp_entry = ttk.Entry(kconf, textvariable=self.temp_var, width=8)
        self.k_temp_entry.grid(row=1, column=5, sticky=tk.W)
        ttk.Label(kconf, text="Top-p").grid(row=1, column=6, sticky=tk.W)
        self.k_topp_entry = ttk.Entry(kconf, textvariable=self.top_p_var, width=8)
        self.k_topp_entry.grid(row=1, column=7, sticky=tk.W)
        ttk.Label(kconf, text="Top-k").grid(row=1, column=8, sticky=tk.W)
        self.k_topk_entry = ttk.Entry(kconf, textvariable=self.top_k_var, width=8)
        self.k_topk_entry.grid(row=1, column=9, sticky=tk.W)
        ttk.Label(kconf, text="Samples").grid(row=1, column=10, sticky=tk.W)
        self.k_samples_entry = ttk.Entry(kconf, textvariable=self.sample_count_var, width=8)
        self.k_samples_entry.grid(row=1, column=11, sticky=tk.W)
        ttk.Button(kconf, text="Apply Kronos", command=self.apply_kronos_config).grid(row=0, column=6, padx=6)

        disp = ttk.Frame(self.root)
        disp.pack(fill=tk.X, padx=8, pady=6)
        ttk.Label(disp, text="Display From").grid(row=0, column=0)
        self.display_start_entry = ttk.Entry(disp, textvariable=self.display_start_var, width=20)
        self.display_start_entry.grid(row=0, column=1)
        ttk.Label(disp, text="Display To").grid(row=0, column=2)
        self.display_end_entry = ttk.Entry(disp, textvariable=self.display_end_var, width=20)
        self.display_end_entry.grid(row=0, column=3)

        self.adv_toggle = ttk.Button(self.root, text="Hide Advanced", command=self.toggle_advanced)
        self.adv_toggle.pack(padx=8, pady=6, anchor=tk.W)

        self.adv = ttk.Frame(self.root)
        self._build_advanced(self.adv)
        self.adv.pack(fill=tk.X)

        actions = ttk.Frame(self.root)
        actions.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(actions, text="Run Forecast", command=self.run_forecast).pack(side=tk.LEFT)
        ttk.Button(actions, text="Visualize Data", command=self.visualize_data).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Clear Plot", command=self.clear_plot).pack(side=tk.LEFT)
        ttk.Button(actions, text="Update Plot", command=self.update_plot).pack(side=tk.LEFT, padx=6)
        ttk.Button(actions, text="Show Table", command=self.show_table).pack(side=tk.LEFT, padx=6)
        ttk.Label(actions, text="Marker Text").pack(side=tk.LEFT)
        ttk.Entry(actions, textvariable=self.marker_text_var, width=18).pack(side=tk.LEFT, padx=4)
        ttk.Label(actions, text="Series").pack(side=tk.LEFT)
        ttk.Combobox(actions, textvariable=self.marker_series_var, values=["Actual","Forecast"], width=10).pack(side=tk.LEFT)
        ttk.Checkbutton(actions, text="Marker Mode", variable=self.marker_mode_var, command=self._toggle_marker_mode).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(actions, text="VLines", variable=self.show_marker_vlines_var, command=lambda: self._render_markers(self.figure.gca() if self.figure else None)).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.root, textvariable=self.status_var).pack(fill=tk.X, padx=8, pady=4)

        plotf = ttk.Frame(self.root)
        plotf.pack(fill=tk.BOTH, expand=True)
        self.plot_frame = plotf
        self.load_default_file()

    def _build_advanced(self, frame):
        g1 = ttk.Frame(frame)
        g1.pack(fill=tk.X, padx=8)
        self.auto_fix_ts_cb = ttk.Checkbutton(g1, text="Auto-fix timestamps (dedupe & fill gaps)", variable=self.auto_fix_ts_var)
        self.auto_fix_ts_cb.grid(row=0, column=0, sticky=tk.W)
        self.full_moon_cb = ttk.Checkbutton(g1, text="Show Full Moon", variable=self.show_full_moon_var)
        self.full_moon_cb.grid(row=0, column=1, sticky=tk.W)
        self.new_moon_cb = ttk.Checkbutton(g1, text="Show New Moon", variable=self.show_new_moon_var)
        self.new_moon_cb.grid(row=0, column=2, sticky=tk.W)
        self.first_quarter_cb = ttk.Checkbutton(g1, text="Show 1st Quarter", variable=self.show_first_quarter_moon_var)
        self.first_quarter_cb.grid(row=0, column=9, sticky=tk.W)
        self.last_quarter_cb = ttk.Checkbutton(g1, text="Show Last Quarter", variable=self.show_last_quarter_moon_var)
        self.last_quarter_cb.grid(row=0, column=10, sticky=tk.W)
        self.earnings_cb = ttk.Checkbutton(g1, text="Show Earnings", variable=self.show_earnings_var, command=self._on_overlay_toggle)
        self.earnings_cb.grid(row=0, column=3, sticky=tk.W)
        self.insider_cb = ttk.Checkbutton(g1, text="Show Insider Sales", variable=self.show_insider_var, command=self._on_overlay_toggle)
        self.insider_cb.grid(row=0, column=4, sticky=tk.W)
        self.insider_proposed_cb = ttk.Checkbutton(g1, text="Show Proposed Insider Sales", variable=self.show_insider_proposed_var, command=self._on_overlay_toggle)
        self.insider_proposed_cb.grid(row=0, column=5, sticky=tk.W)
        self.insider_tooltips_cb = ttk.Checkbutton(g1, text="Insider Tooltips", variable=self.insider_tooltips_var, command=self._on_overlay_toggle)
        self.insider_tooltips_cb.grid(row=0, column=6, sticky=tk.W)
        self.volume_dot_cb = ttk.Checkbutton(g1, text="Volume Dot Chart", variable=self.show_volume_dot_var, command=self._on_overlay_toggle)
        self.volume_dot_cb.grid(row=0, column=7, sticky=tk.W)
        self.insider_proposed_cb = ttk.Checkbutton(g1, text="Show Proposed Insider Sales", variable=self.show_insider_proposed_var, command=self._on_overlay_toggle)
        self.insider_proposed_cb.grid(row=0, column=5, sticky=tk.W)
        self.pe_eps_cb = ttk.Checkbutton(g1, text="Show P/E & EPS", variable=self.show_pe_eps_var, command=self._on_overlay_toggle)
        self.pe_eps_cb.grid(row=0, column=8, sticky=tk.W)

        g1b = ttk.Frame(frame)
        g1b.pack(fill=tk.X, padx=8)
        self.impute_target_cb = ttk.Checkbutton(g1b, text="Impute Missing Target", variable=self.impute_target_var)
        self.impute_target_cb.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(g1b, text="Method").grid(row=0, column=1, sticky=tk.W)
        self.impute_method_cb = ttk.Combobox(g1b, textvariable=self.impute_method_var, values=["ffill","bfill","ffill_bfill","interpolate","interpolate_ffill_bfill"], width=12)
        self.impute_method_cb.grid(row=0, column=2, sticky=tk.W)
        # Duplicate auto_fix_ts checkbox removed

        # Remove TimeGPT finetune controls

        self.add_tooltips()

    def add_tooltip(self, widget, text):
        t = Tooltip(widget, text)
        self._tooltips.append(t)

    def add_tooltips(self):
        self.add_tooltip(self.load_btn, "Load TSV or CSV data file")
        self.add_tooltip(self.symbol_entry, "Stock symbol to fetch via MCP")
        self.add_tooltip(self.load_stock_btn, "Fetch daily OHLCV via MCP")
        self.add_tooltip(self.stock_period_cb, "Length: 1d/5d/1mo/3mo/6mo/1y/2y/5y/max")
        self.add_tooltip(self.k_tokenizer_cb, "HuggingFace Kronos tokenizer repo id")
        self.add_tooltip(self.k_model_cb, "HuggingFace Kronos model repo id")
        self.add_tooltip(self.k_device_cb, "Execution device: cpu or cuda:0")
        self.add_tooltip(self.k_maxctx_entry, "KronosPredictor max context length")
        self.add_tooltip(self.k_clip_entry, "Input clipping range magnitude")
        self.add_tooltip(self.k_temp_entry, "Sampling temperature")
        self.add_tooltip(self.k_topp_entry, "Top-p (nucleus) sampling threshold")
        self.add_tooltip(self.k_topk_entry, "Top-k sampling threshold")
        self.add_tooltip(self.k_samples_entry, "Parallel samples per forecast")
        self.add_tooltip(self.time_col_cb, "Time column name. Parsed as datetime.")
        self.add_tooltip(self.target_col_cb, "Target value column name.")
        self.add_tooltip(self.id_col_cb, "ID column for multiple series. Leave empty for single series.")
        self.add_tooltip(self.series_id_cb, "Select specific series when ID column exists.")
        self.add_tooltip(self.start_entry, "Training start time. Leave empty to use earliest.")
        self.add_tooltip(self.end_entry, "Training end time. Leave empty to use latest.")
        self.add_tooltip(self.freq_entry, "Frequency, e.g., D, H, 15T. Leave empty to infer. For daily stocks, prefer B (Business Day).")
        self.add_tooltip(self.auto_fix_ts_cb, "Ensures continuous timestamps by de-duplicating and filling missing dates.")
        self.add_tooltip(self.impute_target_cb, "Fill missing target values created by the time grid.")
        self.add_tooltip(self.impute_method_cb, "Choose ffill/bfill/ffill_bfill/interpolate for imputation.")
        self.add_tooltip(self.full_moon_cb, "Mark full moon dates; if non-trading, use previous trading day.")
        self.add_tooltip(self.new_moon_cb, "Mark new moon dates; if non-trading, use previous trading day.")
        try:
            self.add_tooltip(self.first_quarter_cb, "Mark first quarter moon dates; adjusts to previous trading day.")
            self.add_tooltip(self.last_quarter_cb, "Mark last quarter moon dates; adjusts to previous trading day.")
        except Exception:
            pass
        self.add_tooltip(self.earnings_cb, "Mark company earnings release dates using the selected symbol.")
        self.add_tooltip(self.insider_cb, "Plot historical insider sale transactions; bubble size reflects transaction value.")
        self.add_tooltip(self.insider_proposed_cb, "Show proposed future insider sales as red dashed v-lines.")
        self.add_tooltip(self.insider_tooltips_cb, "Enable hover tooltips for insider sales with seller and % holdings sold when available.")
        self.add_tooltip(self.volume_dot_cb, "Overlay dots sized by volume; green when close > open else red.")
        self.add_tooltip(self.pe_eps_cb, "Overlay P/E ratio and EPS (TTM) using yfinance/SEC data.")
        self.add_tooltip(self.h_entry, "Forecast horizon h (integer).")
        self.add_tooltip(self.display_start_entry, "Plot start time. Leave empty to auto.")
        self.add_tooltip(self.display_end_entry, "Plot end time. Leave empty to auto.")

    def _to_naive_datetime_index(self, index):
        idx = pd.DatetimeIndex(index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        return idx

    def _extract_eps_row(self, financials):
        if financials is None or len(financials) == 0:
            return None
        candidates = ('Diluted EPS', 'Diluted EPS (USD)', 'Basic EPS', 'Basic EPS (USD)')
        try:
            for label in candidates:
                if label in financials.index:
                    row = financials.loc[label].dropna()
                    if not row.empty:
                        numeric_row = pd.to_numeric(row, errors='coerce').dropna()
                        if not numeric_row.empty:
                            return numeric_row
        except Exception:
            return None
        return None

    def _is_quarter_entry(self, form, frame, fp):
        form = (form or "").upper()
        frame = (frame or "").upper()
        fp = (fp or "").upper()
        if "Q" in frame:
            return True
        if form == "10-Q":
            return True
        if fp.startswith("Q"):
            return True
        return False

    def _try_import_sec_api_cache(self):
        import os, sys, importlib.util
        home = os.path.expanduser("~")
        # Prefer local module in examples
        try:
            here = os.path.join(os.path.dirname(__file__), "sec_api_cache.py")
            if os.path.isfile(here):
                spec = importlib.util.spec_from_file_location("sec_api_cache", here)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        except Exception:
            pass
        candidates = [
            os.path.join(home, "OneDrive", "Documents", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "github", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "windsurf", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "stock_charts_10k-10q0", "stock_charts_10k10q"),
        ]
        for base in candidates:
            try:
                p = os.path.join(base, "sec_api_cache.py")
                if os.path.isfile(p):
                    spec = importlib.util.spec_from_file_location("sec_api_cache", p)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    return mod
            except Exception:
                continue
        try:
            import sec_api_cache
            return sec_api_cache
        except Exception:
            pass
        for p in candidates:
            try:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.append(p)
            except Exception:
                continue
        try:
            import sec_api_cache
            return sec_api_cache
        except Exception:
            return None

    def _fetch_sec_eps_series(self, ticker):
        def _seed_cik_cache(ticker):
            try:
                import shutil, os
                from pathlib import Path
                home = os.path.expanduser("~")
                bases = [
                    os.path.join(home, "OneDrive", "Documents", "stock_charts_10k10q"),
                    os.path.join(home, "OneDrive", "Documents", "github", "stock_charts_10k10q"),
                    os.path.join(home, "OneDrive", "Documents", "windsurf", "stock_charts_10k10q"),
                    os.path.join(home, "OneDrive", "Documents", "stock_charts_10k-10q0", "stock_charts_10k10q"),
                ]
                src = None
                for base in bases:
                    candidate = os.path.join(base, "sec_cache", "cik_lookups", f"{(ticker or '').strip().upper()}.txt")
                    if os.path.isfile(candidate):
                        src = candidate
                        break
                if src:
                    sec_api_cache = self._try_import_sec_api_cache()
                    base_dir = Path(getattr(sec_api_cache, "CACHE_DIR", Path("sec_cache")))
                    dst_dir = base_dir / "cik_lookups"
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    dst = str(dst_dir / f"{(ticker or '').strip().upper()}.txt")
                    shutil.copyfile(src, dst)
                    try:
                        print("Debug: seeded CIK cache from OneDrive", {"src": src, "dst": dst})
                    except Exception:
                        pass
            except Exception:
                pass
        _seed_cik_cache(ticker)
        try:
            import hashlib, json
            from pathlib import Path
            sec_api_cache = self._try_import_sec_api_cache()
            base_dir = Path(getattr(sec_api_cache, "CACHE_DIR", Path("sec_cache")))
            url = "https://www.sec.gov/files/company_tickers.json"
            h = hashlib.md5(url.encode()).hexdigest()
            p = base_dir / "company_tickers" / f"{h}.json"
            dst_dir = base_dir / "cik_lookups"
            if not (dst_dir / f"{(ticker or '').strip().upper()}.txt").exists() and p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                data = payload.get("data")
                cik = None
                if isinstance(data, dict):
                    for _, company in data.items():
                        if str(company.get("ticker", "")).upper() == (ticker or "").strip().upper():
                            cik = str(company.get("cik_str")).zfill(10)
                            break
                elif isinstance(data, list):
                    for rec in data:
                        tk = str(rec.get("ticker", "")).upper()
                        if tk == (ticker or "").strip().upper():
                            cik = str(rec.get("cik_str")).zfill(10)
                            break
                if cik:
                    try:
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        with open(str(dst_dir / f"{(ticker or '').strip().upper()}.txt"), "w") as wf:
                            wf.write(cik)
                        print("Debug: seeded CIK from cached company_tickers", {"cik": cik})
                    except Exception:
                        pass
                else:
                    try:
                        mapping = {
                            "AAPL": "0000320193",
                            "AMD": "0000002488",
                            "COST": "0000909832",
                        }
                        tk = (ticker or "").strip().upper()
                        val = mapping.get(tk)
                        if val and not (dst_dir / f"{tk}.txt").exists():
                            dst_dir.mkdir(parents=True, exist_ok=True)
                            with open(str(dst_dir / f"{tk}.txt"), "w") as wf:
                                wf.write(val)
                            print("Debug: seeded CIK from fallback mapping", {"ticker": tk, "cik": val})
                    except Exception:
                        pass
        except Exception:
            pass
        sec_api_cache = self._try_import_sec_api_cache()
        if sec_api_cache is None:
            print("Debug: SEC EPS required; sec_api_cache not available")
            return None
        try:
            from pathlib import Path
            home = os.path.expanduser("~")
            bases = [
                os.path.join(home, "OneDrive", "Documents", "windsurf", "stock_charts_10k10q"),
                os.path.join(home, "OneDrive", "Documents", "stock_charts_10k10q"),
                os.path.join(home, "OneDrive", "Documents", "github", "stock_charts_10k10q"),
                os.path.join(home, "OneDrive", "Documents", "stock_charts_10k-10q0", "stock_charts_10k10q"),
            ]
            for base in bases:
                sec_dir = os.path.join(base, "sec_cache")
                if os.path.isdir(sec_dir):
                    sec_api_cache.CACHE_DIR = Path(sec_dir)
                    try:
                        print("Debug: using external SEC cache dir", {"dir": sec_dir})
                    except Exception:
                        pass
                    break
        except Exception:
            pass
        try:
            # Inject in-memory company tickers cache to bypass network if CIK file exists
            from pathlib import Path
            base_dir = Path(getattr(sec_api_cache, "CACHE_DIR"))
            f = base_dir / "cik_lookups" / f"{(ticker or '').strip().upper()}.txt"
            if f.exists():
                with open(f, "r") as rf:
                    cik_val = rf.read().strip()
                if cik_val:
                    sec_api_cache.company_tickers_cache = {
                        "0": {"ticker": (ticker or "").strip().upper(), "cik_str": int(cik_val), "title": (ticker or "").strip().upper()}
                    }
                    from datetime import datetime as _dt
                    sec_api_cache.company_tickers_last_update = _dt.now()
                    print("Debug: injected in-memory company tickers cache", {"ticker": (ticker or '').strip().upper(), "cik": cik_val})
        except Exception:
            pass
        cik = None
        try:
            cik = sec_api_cache.get_company_cik((ticker or "").strip().upper())
            print("Debug: sec_api_cache CIK", {"symbol": (ticker or "").strip().upper(), "cik": cik})
        except Exception:
            cik = None
        if not cik:
            return None
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        data = None
        try:
            data = sec_api_cache.make_sec_request(url, "company_facts")
            print("Debug: SEC companyfacts fetched", {"symbol": (ticker or "").strip().upper(), "has_data": isinstance(data, dict)})
        except Exception:
            data = None
        if not isinstance(data, dict):
            return None
        facts = data.get("facts") or {}
        candidates = [
            ("us-gaap", "EarningsPerShareDiluted"),
            ("us-gaap", "EarningsPerShareBasic"),
            ("us-gaap", "DilutedEarningsPerShare"),
            ("ifrs-full", "EarningsPerShareDiluted"),
            ("ifrs-full", "EarningsPerShareBasic"),
        ]
        eps_fact = None
        for taxonomy, concept in candidates:
            taxonomy_bucket = facts.get(taxonomy)
            if taxonomy_bucket and concept in taxonomy_bucket:
                eps_fact = taxonomy_bucket[concept]
                break
        if eps_fact is None:
            for taxonomy_bucket in facts.values():
                if isinstance(taxonomy_bucket, dict):
                    for concept in ("EarningsPerShareDiluted", "EarningsPerShareBasic"):
                        if concept in taxonomy_bucket:
                            eps_fact = taxonomy_bucket[concept]
                            break
                if eps_fact:
                    break
        if eps_fact is None:
            return None
        units = eps_fact.get("units") or {}
        quarterly_entries = []
        annual_entries = []
        for values in units.values():
            for item in values:
                val = item.get("val")
                end = item.get("end") or item.get("filed")
                if val is None or end is None:
                    continue
                try:
                    dt = pd.Timestamp(end)
                except Exception:
                    continue
                form = item.get("form")
                frame = item.get("frame")
                fp = item.get("fp")
                if self._is_quarter_entry(form, frame, fp):
                    quarterly_entries.append((dt, val))
                else:
                    annual_entries.append((dt, val))
        try:
            print("Debug: SEC EPS entries", {"quarterly": len(quarterly_entries), "annual": len(annual_entries)})
        except Exception:
            pass
        segments = []
        if quarterly_entries:
            quarterly_entries.sort(key=lambda x: x[0])
            q_idx = pd.DatetimeIndex([dt for dt, _ in quarterly_entries])
            q_vals = [val for _, val in quarterly_entries]
            q_series = pd.Series(q_vals, index=self._to_naive_datetime_index(q_idx))
            q_series = q_series[~q_series.index.duplicated(keep='last')].sort_index()
            q_ttm = q_series.rolling(window=4, min_periods=4).sum().dropna()
            if not q_ttm.empty:
                segments.append(q_ttm)
        if annual_entries:
            annual_entries.sort(key=lambda x: x[0])
            a_idx = pd.DatetimeIndex([dt for dt, _ in annual_entries])
            a_vals = [val for _, val in annual_entries]
            a_series = pd.Series(a_vals, index=self._to_naive_datetime_index(a_idx))
            a_series = a_series[~a_series.index.duplicated(keep='last')].sort_index()
            if not a_series.empty:
                segments.append(a_series)
            try:
                print("Debug: SEC annual EPS series", {"points": int(len(a_series)), "start": str(a_series.index.min()), "end": str(a_series.index.max())})
            except Exception:
                pass
        else:
            try:
                print("Debug: SEC annual EPS series", {"points": 0})
            except Exception:
                pass
        if not segments:
            return None
        combined = pd.concat(segments)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()

    def _compute_eps_series(self, symbol):
        eps_segments = []
        sec_eps = None
        try:
            sec_eps = self._fetch_sec_eps_series(symbol)
        except Exception:
            sec_eps = None
        if sec_eps is None or getattr(sec_eps, 'empty', True):
            print("Debug: SEC EPS required; skipping yfinance-only fallback")
            return None
        eps_segments.append(sec_eps)
        combined = pd.concat(eps_segments)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined = combined.sort_index()
        try:
            print("Debug: EPS series merged", {"total_points": int(len(combined)), "start": str(combined.index.min()), "end": str(combined.index.max())})
        except Exception:
            pass
        return combined

    def _plot_signed_line(self, ax, x, y, label, pos_color='green', neg_color='red', linewidth=1.8):
        import numpy as np
        ys = pd.Series(pd.to_numeric(y, errors='coerce'))
        xp = pd.Series(x)
        y_pos = ys.where(ys > 0, np.nan)
        y_neg = ys.where(ys <= 0, np.nan)
        line_pos = None
        line_neg = None
        if not y_pos.isna().all():
            line_pos, = ax.plot(xp, y_pos, linewidth=linewidth, color=pos_color, label=f"{label} (+)")
        if not y_neg.isna().all():
            line_neg, = ax.plot(xp, y_neg, linewidth=linewidth, color=neg_color, label=f"{label} (-)")
        return line_pos, line_neg

    def _plot_pe_eps_overlay(self, ax, df, time_col, target_col, symbol):
        if ax is None or df is None or df.empty or not time_col:
            return
        try:
            eps_series = self._compute_eps_series(symbol)
        except Exception:
            eps_series = None
        if eps_series is None or eps_series.empty:
            try:
                self.status_var.set("P/E & EPS: no EPS series available")
            except Exception:
                pass
            return
        dfx = df.copy()
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
        dfx = dfx.dropna(subset=[time_col])
        dfx = dfx.sort_values(time_col)
        candidates_close = [target_col, 'close', 'Close', 'Adj Close']
        close_col = next((c for c in candidates_close if c and c in dfx.columns), None)
        if close_col is None:
            return
        dfx['__EPS__'] = pd.Series(index=dfx.index, dtype=float)
        for date, eps in eps_series.items():
            try:
                dfx.loc[dfx[time_col] >= pd.to_datetime(date), '__EPS__'] = float(eps)
            except Exception:
                continue
        dfx['__EPS__'] = dfx['__EPS__'].ffill()
        dfx['__PE__'] = pd.to_numeric(dfx[close_col], errors='coerce') / pd.to_numeric(dfx['__EPS__'], errors='coerce')
        dfx = dfx.dropna(subset=['__PE__', '__EPS__'])
        if dfx.empty:
            return
        x = dfx[time_col]
        ax2 = ax.twinx()
        pe_pos, pe_neg = self._plot_signed_line(ax2, x, dfx['__PE__'], 'P/E Ratio')
        ax2.set_ylabel('P/E', color=(pe_pos.get_color() if pe_pos else (pe_neg.get_color() if pe_neg else '#1f77b4')))
        ax2.tick_params(axis='y', labelcolor=(pe_pos.get_color() if pe_pos else (pe_neg.get_color() if pe_neg else '#1f77b4')))
        ax3 = ax.twinx()
        try:
            ax3.spines['right'].set_position(('axes', 1.07))
        except Exception:
            pass
        eps_pos, eps_neg = self._plot_signed_line(ax3, x, dfx['__EPS__'], 'EPS (TTM)', linewidth=1.6)
        eps_lbl_color = (eps_pos.get_color() if eps_pos else (eps_neg.get_color() if eps_neg else '#2ca02c'))
        ax3.set_ylabel('EPS', color=eps_lbl_color)
        ax3.tick_params(axis='y', labelcolor=eps_lbl_color)
        try:
            latest_eps = dfx['__EPS__'].iloc[-1]
            ann_color = ('green' if float(latest_eps) > 0 else 'red')
            ax3.annotate(f'{latest_eps:.2f}', xy=(x.iloc[-1], latest_eps), xytext=(-10, 10), textcoords='offset points', ha='right', fontsize=9, color=ann_color)
        except Exception:
            pass
        try:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            h3, l3 = ax3.get_legend_handles_labels()
            ax.legend(h1 + h2 + h3, l1 + l2 + l3, loc='best')
        except Exception:
            pass
        try:
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.draw()
        except Exception:
            pass

    def _plot_pe_eps_two_panel(self, ax_pe, ax_eps, df, time_col, target_col, symbol):
        if ax_pe is None or ax_eps is None or df is None or df.empty or not time_col:
            return None, None
        try:
            eps_series = self._compute_eps_series(symbol)
        except Exception:
            eps_series = None
        if eps_series is None or eps_series.empty:
            try:
                self.status_var.set("P/E & EPS: no EPS series available")
            except Exception:
                pass
            return None, None
        dfx = df.copy()
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
        dfx = dfx.dropna(subset=[time_col])
        dfx = dfx.sort_values(time_col)
        candidates_close = [target_col, 'close', 'Close', 'Adj Close']
        close_col = next((c for c in candidates_close if c and c in dfx.columns), None)
        if close_col is None:
            return None, None
        dfx['__EPS__'] = pd.Series(index=dfx.index, dtype=float)
        for date, eps in eps_series.items():
            try:
                dfx.loc[dfx[time_col] >= pd.to_datetime(date), '__EPS__'] = float(eps)
            except Exception:
                continue
        dfx['__EPS__'] = dfx['__EPS__'].ffill()
        dfx['__PE__'] = pd.to_numeric(dfx[close_col], errors='coerce') / pd.to_numeric(dfx['__EPS__'], errors='coerce')
        dfx = dfx.dropna(subset=['__PE__', '__EPS__'])
        if dfx.empty:
            return None, None
        try:
            first_idx = dfx['__EPS__'].first_valid_index()
            first_dt = dfx.loc[first_idx, time_col] if first_idx is not None else None
            print("Debug: overlay EPS mapping", {"first_eps_date": str(pd.to_datetime(first_dt)) if first_dt is not None else None, "mapped_points": int(len(dfx.dropna(subset=['__EPS__'])))})
        except Exception:
            pass
        x = dfx[time_col]
        pe_pos, pe_neg = self._plot_signed_line(ax_pe, x, dfx['__PE__'], 'Trailing P/E Ratio', linewidth=2)
        pe_lbl_color = (pe_pos.get_color() if pe_pos else (pe_neg.get_color() if pe_neg else '#1f77b4'))
        ax_pe.set_ylabel('P/E Ratio', color=pe_lbl_color)
        ax_pe.tick_params(axis='y', labelcolor=pe_lbl_color)
        ax_pe.grid(True, alpha=0.3)
        ax_price = ax_pe.twinx()
        price_line, = ax_price.plot(x, dfx[close_col], linewidth=1.5, label='Close Price', color='#ff7f0e')
        ax_price.set_ylabel('Close Price', color=price_line.get_color())
        ax_price.tick_params(axis='y', labelcolor=price_line.get_color())
        lines = [l for l in [pe_pos, pe_neg, price_line] if l is not None]
        try:
            ax_pe.legend(lines, [line.get_label() for line in lines], loc='upper left')
        except Exception:
            pass
        try:
            ax_pe.set_title(f'{symbol} PE & Price Over Time')
        except Exception:
            pass
        eps_pos, eps_neg = self._plot_signed_line(ax_eps, x, dfx['__EPS__'], 'Annualized EPS (TTM)', linewidth=1.8)
        ax_eps.set_ylabel('EPS (TTM)')
        ax_eps.set_xlabel('Date')
        ax_eps.grid(True, alpha=0.3)
        try:
            latest_eps = dfx['__EPS__'].iloc[-1]
            ann_color = ('green' if float(latest_eps) > 0 else 'red')
            ax_eps.annotate(f'Latest: {latest_eps:.2f}', xy=(x.iloc[-1], latest_eps), xytext=(-10, 10), textcoords='offset points', ha='right', fontsize=9, color=ann_color)
        except Exception:
            pass
        try:
            ax_eps.legend(loc='upper left')
        except Exception:
            pass
        return dfx, close_col

    def _load_api_key(self) -> str:
        key = (os.getenv("NIXTLA_API_KEY", "") or "").strip()
        if key:
            return key
        candidates = []
        try:
            candidates.append(os.path.join(os.getcwd(), ".env"))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.dirname(__file__), ".env"))
        except Exception:
            pass
        try:
            candidates.append(os.path.expanduser("~/.env"))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.expanduser("~"), ".nixtla", "config.json"))
        except Exception:
            pass
        try:
            candidates.append(os.path.join(os.path.dirname(__file__), "nixtla_config.json"))
        except Exception:
            pass
        for p in candidates:
            try:
                if os.path.isfile(p):
                    if p.endswith(".json"):
                        with open(p, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        v = (str(data.get("NIXTLA_API_KEY", "")) or str(data.get("api_key", ""))).strip()
                        if v:
                            return v
                    else:
                        with open(p, "r", encoding="utf-8") as f:
                            for line in f:
                                s = line.strip()
                                if s.startswith("NIXTLA_API_KEY="):
                                    v = s.split("=", 1)[1].strip().strip('"').strip("'")
                                    if v:
                                        return v
            except Exception:
                pass
        return ""

    def toggle_advanced(self):
        if self.advanced_visible:
            self.adv.pack_forget()
            self.advanced_visible = False
            try:
                self.adv_toggle.configure(text="Show Advanced")
            except Exception:
                pass
        else:
            self.adv.pack(fill=tk.X)
            self.advanced_visible = True
            try:
                self.adv_toggle.configure(text="Hide Advanced")
            except Exception:
                pass

    def load_file(self, path=None):
        if not path:
            path = filedialog.askopenfilename(filetypes=[("TSV files","*.tsv"), ("CSV files","*.csv"), ("All files","*.*")])
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.df = df
        time_col, target_col, id_col = self._infer_columns(df)
        if not id_col:
            base = os.path.splitext(os.path.basename(path))[0]
            self.df["unique_id"] = base
            id_col = "unique_id"
        if time_col:
            self.time_col_var.set(time_col)
        if target_col:
            self.target_col_var.set(target_col)
        if id_col:
            self.id_col_var.set(id_col)
        self.file_path_var.set(path)
        self.update_columns()
        self.status_var.set(f"Loaded {os.path.basename(path)} ({len(self.df)} rows)")

    def load_stock_data(self):
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showwarning("Warning", "Enter a stock symbol")
            return
        period = self.stock_period_var.get().strip() or "1y"
        try:
            import os
            import asyncio
            try:
                from spoonos_stock_agent import MCPClient
            except ImportError:
                candidates = [
                    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spoon-core-jue")),
                    r"C:\\Users\\juesh\\jules\\spoon-core-jue",
                    os.path.expanduser(r"~\\jules\\spoon-core-jue"),
                ]
                for p in candidates:
                    if os.path.isdir(p) and p not in sys.path:
                        sys.path.append(p)
                from spoonos_stock_agent import MCPClient
            async def fetch():
                client = MCPClient(server_name="stock-mcp")
                return await client.call_tool("get_stock_historical_data", {"symbol": symbol, "period": period, "interval": "1d"})
            res = asyncio.run(fetch())
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if isinstance(res, dict) and "error" in res:
            messagebox.showerror("Error", res.get("error", "Unknown error"))
            return
        try:
            data = res.get("data", []) if isinstance(res, dict) else []
            if not data:
                messagebox.showerror("Error", "No data returned from MCP")
                return
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            target_col = "adj_close" if "adj_close" in df.columns else "close"
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            df = df.dropna(subset=["date", target_col])
            df["unique_id"] = symbol
            self.df = df
            self.file_path_var.set(f"MCP:{symbol} {period} 1d")
            self.id_col_var.set("unique_id")
            self.time_col_var.set("date")
            self.target_col_var.set(target_col)
            self.update_columns()
            self.status_var.set(f"Loaded {symbol} ({len(self.df)} rows) via MCP")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

    def load_default_file(self):
        default_path = r".\data\es_4h.csv"
        if os.path.exists(default_path):
            try:
                self.load_file(default_path)
            except Exception:
                pass

    def update_columns(self):
        if self.df is None:
            return
        cols = list(self.df.columns)
        self.time_col_cb["values"] = cols
        self.target_col_cb["values"] = cols
        self.id_col_cb["values"] = [""] + cols
        if self.id_col_var.get() in cols:
            ids = sorted(self.df[self.id_col_var.get()].dropna().unique().tolist())
        else:
            ids = []
        self.series_id_cb["values"] = ids
        if len(ids) > 0:
            self.series_id_var.set(ids[0])
        if self.time_col_var.get() in cols:
            try:
                s = pd.to_datetime(self.df[self.time_col_var.get()], errors="coerce")
                s_valid = s.dropna()
                if not s_valid.empty:
                    self.start_var.set(str(s_valid.min()))
                    self.end_var.set(str(s_valid.max()))
                    sf = self._suggest_freq(s_valid)
                    if sf:
                        self.freq_var.set(sf)
            except Exception:
                pass

    def use_full_range(self):
        if self.df is None or self.time_col_var.get() == "":
            return
        try:
            s = pd.to_datetime(self.df[self.time_col_var.get()], errors="coerce")
            s_valid = s.dropna()
            if not s_valid.empty:
                self.start_var.set(str(s_valid.min()))
                self.end_var.set(str(s_valid.max()))
        except Exception:
            pass

    def _suggest_freq(self, s: pd.Series):
        try:
            s_sorted = s.sort_values()
            inf = pd.infer_freq(s_sorted)
            if inf:
                return inf
            diffs = s_sorted.diff().dropna()
            if not diffs.empty:
                mode = diffs.mode().iloc[0]
                days = getattr(mode, 'days', None)
                if days == 1:
                    weekday_counts = s_sorted.dt.weekday.value_counts()
                    if weekday_counts.get(5, 0) == 0 and weekday_counts.get(6, 0) == 0:
                        return 'B'
                    return 'D'
            return None
        except Exception:
            return None

    def _infer_columns(self, df: pd.DataFrame):
        cols = list(df.columns)
        time_candidates = ["date", "ds", "time", "timestamp", "datetime"]
        target_candidates = ["adj_close", "close", "y", "value", "price"]
        id_candidates = ["unique_id", "id", "symbol", "ticker", "series"]
        time_col = None
        for c in time_candidates:
            if c in cols:
                time_col = c
                break
        if not time_col:
            for c in cols:
                try:
                    s = pd.to_datetime(df[c], errors="coerce")
                    if s.notna().mean() > 0.6:
                        time_col = c
                        break
                except Exception:
                    pass
        id_col = None
        for c in id_candidates:
            if c in cols:
                id_col = c
                break
        target_col = None
        for c in target_candidates:
            if c in cols:
                target_col = c
                break
        if not target_col:
            numeric_cols = []
            for c in cols:
                if c == time_col or c == id_col:
                    continue
                try:
                    s = pd.to_numeric(df[c], errors="coerce")
                    if s.notna().mean() > 0.6:
                        numeric_cols.append((c, s.notna().mean()))
                except Exception:
                    pass
            if numeric_cols:
                numeric_cols.sort(key=lambda x: -x[1])
                target_col = numeric_cols[0][0]
        return time_col, target_col, id_col

    def _auto_fix_timestamps(self, df: pd.DataFrame, id_col: str | None, time_col: str, target_col: str, freq: str | None) -> pd.DataFrame:
        if id_col and id_col in df.columns:
            df = df.drop_duplicates(subset=[id_col, time_col], keep='last')
        else:
            df = df.drop_duplicates(subset=[time_col], keep='last')
        if not freq:
            return df
        def grid_group(g: pd.DataFrame) -> pd.DataFrame:
            idx = pd.date_range(start=g[time_col].min(), end=g[time_col].max(), freq=freq)
            g2 = g.set_index(time_col).reindex(idx)
            g2.index.name = time_col
            g2 = g2.reset_index()
            if id_col and id_col in g.columns:
                g2[id_col] = g[id_col].iloc[0]
            return g2
        if id_col and id_col in df.columns:
            out = (
                df.groupby(id_col, observed=True, sort=False, group_keys=False)
                .apply(grid_group)
                .reset_index(drop=True)
            )
        else:
            out = grid_group(df)
        return out

    def _supports_multivariate(self, model: str) -> bool:
        # Default TimeGPT models in this GUI do not support multivariate.
        return False

    def on_model_change(self):
        model = self.model_var.get()
        if not self._supports_multivariate(model):
            try:
                self.multivariate_var.set(False)
                self.multivariate_cb.state(["disabled"])
            except Exception:
                pass
        else:
            try:
                self.multivariate_cb.state(["!disabled"])
            except Exception:
                pass

    def _impute_target(self, df: pd.DataFrame, id_col: str | None, time_col: str, target_col: str, method: str) -> pd.DataFrame:
        if method == "ffill":
            if id_col and id_col in df.columns:
                df[target_col] = df.groupby(id_col, observed=True)[target_col].transform(lambda s: s.ffill())
            else:
                df[target_col] = df[target_col].ffill()
        elif method == "bfill":
            if id_col and id_col in df.columns:
                df[target_col] = df.groupby(id_col, observed=True)[target_col].transform(lambda s: s.bfill())
            else:
                df[target_col] = df[target_col].bfill()
        elif method == "ffill_bfill":
            if id_col and id_col in df.columns:
                df[target_col] = df.groupby(id_col, observed=True)[target_col].transform(lambda s: s.ffill().bfill())
            else:
                df[target_col] = df[target_col].ffill().bfill()
        elif method == "interpolate" or method == "interpolate_ffill_bfill":
            if id_col and id_col in df.columns:
                interp = df.groupby(id_col, observed=True).apply(lambda g: g.set_index(time_col)[target_col].interpolate(method="time")).reset_index(level=0, drop=True)
                df[target_col] = interp
            else:
                df = df.set_index(time_col)
                df[target_col] = df[target_col].interpolate(method="time")
                df = df.reset_index()
            if method == "interpolate_ffill_bfill":
                if id_col and id_col in df.columns:
                    df[target_col] = df.groupby(id_col, observed=True)[target_col].transform(lambda s: s.ffill().bfill())
                else:
                    df[target_col] = df[target_col].ffill().bfill()
        return df

    def _parse_levels(self):
        txt = self.level_var.get().strip()
        if not txt:
            return None
        try:
            return [float(x) for x in txt.split(",") if x.strip()]
        except Exception:
            return None

    def _parse_quantiles(self):
        txt = self.quantiles_var.get().strip()
        if not txt:
            return None
        try:
            return [float(x) for x in txt.split(",") if x.strip()]
        except Exception:
            return None

    def visualize_data(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load a TSV file first")
            return
        id_col = self.id_col_var.get() if self.id_col_var.get() else None
        time_col = self.time_col_var.get()
        target_col = self.target_col_var.get()
        if not time_col or not target_col:
            messagebox.showwarning("Warning", "Select time and target columns")
            return

        try:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            messagebox.showerror("Error", "matplotlib is required: " + str(e))
            return

        self.clear_plot()
        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            from matplotlib.figure import Figure
            fig = Figure(figsize=(12, 8), dpi=100)
            gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1])
            ax_pe = fig.add_subplot(gs[0])
            ax_eps = fig.add_subplot(gs[1], sharex=ax_pe)
        else:
            fig = Figure(figsize=(9, 5), dpi=100)
            ax = fig.add_subplot(111)

        # Prepare actual data
        act_df = self.df.copy()
        act_df[time_col] = pd.to_datetime(act_df[time_col], errors="coerce")
        act_df[target_col] = pd.to_numeric(act_df[target_col], errors="coerce")
        act_df = act_df.dropna(subset=[time_col, target_col])

        if id_col in act_df.columns and self.series_id_var.get():
            act_df = act_df[act_df[id_col] == self.series_id_var.get()]

        act_df = act_df.sort_values(time_col)

        # Apply Display filters
        disp_start = self.display_start_var.get().strip()
        disp_end = self.display_end_var.get().strip()

        if disp_start:
            ds = pd.to_datetime(disp_start, errors="coerce")
            if pd.notna(ds):
                act_df = act_df[act_df[time_col] >= ds]
        if disp_end:
            de = pd.to_datetime(disp_end, errors="coerce")
            if pd.notna(de):
                act_df = act_df[act_df[time_col] <= de]

        if act_df.empty:
             messagebox.showinfo("Info", "No data to display in the selected range")
             return

        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            sym = self.symbol_var.get().strip()
            if not sym and 'unique_id' in act_df.columns:
                vals = [str(v).strip() for v in act_df['unique_id'].dropna().tolist() if str(v).strip()]
                sym = vals[0] if vals else sym
            dfx, close_col = self._plot_pe_eps_two_panel(ax_pe, ax_eps, act_df, time_col, (target_col if target_col in act_df.columns else 'close'), sym)
        else:
            ax.plot(act_df[time_col], act_df[target_col], label="Actual", color="#1f77b4")
            ax.legend(loc="best")
            ax.set_title(f"Data Visualization: {self.series_id_var.get() if self.series_id_var.get() else 'All Series'}")
            ax.set_xlabel(time_col)
            ax.set_ylabel(target_col)

        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.status_var.set("Data visualization complete")
        try:
            if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                if dfx is not None and close_col is not None:
                    self._setup_hover(ax_pe, [(pd.Series(dfx[time_col]), pd.Series(dfx[close_col]), "Actual")])
            else:
                x = act_df[time_col]
                y = act_df[target_col]
                self._setup_hover(ax, [(x, y, "Actual")])
        except Exception:
            pass
        self.last_act_df = act_df
        self.last_act_time_col = time_col
        self.last_act_target = target_col
        try:
            if bool(self.show_full_moon_var.get()) or bool(self.show_new_moon_var.get()) or bool(self.show_first_quarter_moon_var.get()) or bool(self.show_last_quarter_moon_var.get()):
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_moon_markers(ax_pe, act_df, time_col, (target_col if target_col in act_df.columns else 'close'), bool(self.show_full_moon_var.get()), bool(self.show_new_moon_var.get()), bool(self.show_first_quarter_moon_var.get()), bool(self.show_last_quarter_moon_var.get()))
                else:
                    self._plot_moon_markers(ax, act_df, time_col, target_col, bool(self.show_full_moon_var.get()), bool(self.show_new_moon_var.get()), bool(self.show_first_quarter_moon_var.get()), bool(self.show_last_quarter_moon_var.get()))
        except Exception:
            pass
        try:
            if hasattr(self, 'show_earnings_var') and bool(self.show_earnings_var.get()):
                try:
                    print("Debug: visualizing earnings overlay", {"symbol": self.symbol_var.get().strip(), "rows": len(act_df)})
                except Exception:
                    pass
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_earnings_markers(ax_pe, act_df, time_col, (target_col if target_col in act_df.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_earnings_markers(ax, act_df, time_col, target_col, self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_insider_var') and bool(self.show_insider_var.get()):
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_insider_sales(ax_pe, act_df, time_col, (target_col if target_col in act_df.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_insider_sales(ax, act_df, time_col, target_col, self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_insider_proposed_var') and bool(self.show_insider_proposed_var.get()):
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_insider_proposed_sales(ax_pe, act_df, time_col, (target_col if target_col in act_df.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_insider_proposed_sales(ax, act_df, time_col, target_col, self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_volume_dot_var') and bool(self.show_volume_dot_var.get()):
                try:
                    print("Debug: visualizing volume dot overlay", {"symbol": self.symbol_var.get().strip(), "rows": len(act_df)})
                except Exception:
                    pass
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_volume_dot_overlay(ax_pe, act_df, time_col, (target_col if target_col in act_df.columns else 'close'))
                else:
                    self._plot_volume_dot_overlay(ax, act_df, time_col, (target_col if target_col in act_df.columns else 'close'))
                try:
                    if hasattr(self, 'canvas') and self.canvas:
                        self.canvas.draw()
                except Exception:
                    pass
        except Exception:
            pass
        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            self._render_markers(ax_pe)
        else:
            self._render_markers(ax)

    def run_forecast(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load a TSV file first")
            return
        id_col = self.id_col_var.get() if self.id_col_var.get() else None
        time_col = self.time_col_var.get()
        target_col = self.target_col_var.get() or "close"
        if not time_col:
            messagebox.showwarning("Warning", "Select time column")
            return
        if self.kronos_predictor is None:
            messagebox.showerror("Error", "Kronos model/tokenizer failed to load")
            return
        try:
            df = self.df.copy()
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.dropna(subset=[time_col])
            if id_col and id_col in df.columns and self.series_id_var.get():
                df = df[df[id_col] == self.series_id_var.get()]
            s = df[time_col]
            start = self.start_var.get().strip()
            end = self.end_var.get().strip()
            if start:
                df = df[s >= pd.to_datetime(start)]
            if end:
                df = df[s <= pd.to_datetime(end)]
            df = df.sort_values(time_col)
            freq = self.freq_var.get().strip() or self._suggest_freq(df[time_col].dropna()) or 'B'
            if self.auto_fix_ts_var.get():
                df = self._auto_fix_timestamps(df, id_col, time_col, target_col, freq)
            # Ensure required columns
            price_cols = ['open','high','low','close']
            for c in price_cols:
                if c not in df.columns and target_col in df.columns:
                    df[c] = pd.to_numeric(df[target_col], errors="coerce")
            if 'close' not in df.columns and target_col in df.columns:
                df['close'] = pd.to_numeric(df[target_col], errors="coerce")
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            if 'amount' not in df.columns:
                df['amount'] = df['close'] * df['volume']
            # Drop NA in required cols
            req_cols = price_cols + ['volume','amount']
            df = df.dropna(subset=[time_col] + [c for c in req_cols if c in df.columns])
            if df.empty:
                messagebox.showerror("Error", "No usable rows after cleaning")
                return
            self.filtered_df = df
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        # Build windows
        try:
            h = int(self.h_var.get())
        except Exception:
            h = 24
        lookback = min(400, max(10, len(self.filtered_df)))
        hist = self.filtered_df.tail(lookback)
        x_df = hist[['open','high','low','close','volume','amount']]
        x_timestamp = hist[time_col]
        try:
            step = pd.tseries.frequencies.to_offset(freq)
        except Exception:
            step = pd.tseries.frequencies.to_offset('B')
        start_future = pd.to_datetime(self.filtered_df[time_col].max()) + step
        y_timestamp = pd.Series(pd.date_range(start=start_future, periods=h, freq=freq))
        # Predict via Kronos
        try:
            try:
                device_req = self.kronos_device_var.get().strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
                if self.kronos_predictor and device_req != getattr(self, "_kronos_device", device_req):
                    self.kronos_predictor.tokenizer = self.kronos_predictor.tokenizer.to(device_req)
                    self.kronos_predictor.model = self.kronos_predictor.model.to(device_req)
                    self.kronos_predictor.device = device_req
                    self._kronos_device = device_req
                if self.kronos_predictor:
                    self.kronos_predictor.max_context = int(self.max_context_var.get())
                    self.kronos_predictor.clip = int(self.clip_var.get())
            except Exception:
                pass
            pred_df = self.kronos_predictor.predict(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=h,
                T=float(self.temp_var.get()),
                top_k=int(self.top_k_var.get()),
                top_p=float(self.top_p_var.get()),
                sample_count=int(self.sample_count_var.get()),
                verbose=True,
            )
            self.last_fcst_df = pred_df
            self.last_id_col = id_col or "unique_id"
            self.last_time_col = time_col
            self.last_target_col = 'close'
            self.last_levels = None
            self.last_quantiles = None
            self.plot_results(pred_df, self.last_id_col, self.last_time_col, self.last_target_col, None, None)
            self.status_var.set("Forecast complete (Kronos)")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

    def apply_kronos_config(self):
        tid = self.kronos_tokenizer_id_var.get().strip() or "NeoQuasar/Kronos-Tokenizer-base"
        mid = self.kronos_model_id_var.get().strip() or "NeoQuasar/Kronos-small"
        device = self.kronos_device_var.get().strip() or ("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            tok = KronosTokenizer.from_pretrained(tid)
            mod = Kronos.from_pretrained(mid)
            pred = KronosPredictor(mod, tok, device=device, max_context=int(self.max_context_var.get()), clip=int(self.clip_var.get()))
            self.kronos_tokenizer = tok
            self.kronos_model = mod
            self.kronos_predictor = pred
            self._kronos_device = device
            self.status_var.set("Kronos config applied")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_plot(self):
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.figure = None

    def plot_results(self, fcst_df, id_col, time_col, target_col, levels, quantiles):
        try:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        except Exception as e:
            messagebox.showerror("Error", "matplotlib is required: " + str(e))
            return
        self.clear_plot()
        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            fig = Figure(figsize=(12, 8), dpi=100)
            gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1])
            ax_pe = fig.add_subplot(gs[0])
            ax_eps = fig.add_subplot(gs[1], sharex=ax_pe)
        else:
            fig = Figure(figsize=(9, 5), dpi=100)
            ax = fig.add_subplot(111)
        act_df = self.df.copy()
        act_df[time_col] = pd.to_datetime(act_df[time_col], errors="coerce")
        act_df[target_col] = pd.to_numeric(act_df[target_col], errors="coerce") if target_col in act_df.columns else act_df.get('close')
        act_df = act_df.dropna(subset=[time_col])
        if id_col in act_df.columns and self.series_id_var.get():
            act_df = act_df[act_df[id_col] == self.series_id_var.get()]
        fcst_df = fcst_df.copy()
        fcst_df[time_col] = pd.to_datetime(fcst_df.index if time_col not in fcst_df.columns else fcst_df[time_col], errors="coerce")
        disp_start = self.display_start_var.get().strip()
        disp_end = self.display_end_var.get().strip()
        fcst_end = fcst_df[time_col].max() if time_col in fcst_df.columns else pd.to_datetime(fcst_df.index).max()
        act_ext = act_df[act_df[time_col] <= fcst_end]
        if disp_start:
            ds = pd.to_datetime(disp_start, errors="coerce")
            if pd.notna(ds):
                act_ext = act_ext[act_ext[time_col] >= ds]
        if disp_end:
            de = pd.to_datetime(disp_end, errors="coerce")
            if pd.notna(de):
                act_ext = act_ext[act_ext[time_col] <= de]
        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            sym = self.symbol_var.get().strip()
            if not sym and 'unique_id' in act_ext.columns:
                vals = [str(v).strip() for v in act_ext['unique_id'].dropna().tolist() if str(v).strip()]
                sym = vals[0] if vals else sym
            dfx, close_col = self._plot_pe_eps_two_panel(ax_pe, ax_eps, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), sym)
            y_time = fcst_df[time_col] if time_col in fcst_df.columns else pd.to_datetime(fcst_df.index)
            if 'close' in fcst_df.columns:
                ax_pe.plot(y_time, fcst_df['close'], linestyle="-", color="#ff7f0e", label="Forecast (close)")
            try:
                handles, labels = ax_pe.get_legend_handles_labels()
                ax_pe.legend(handles, labels, loc="best")
            except Exception:
                pass
        else:
            if target_col in act_ext.columns:
                ax.plot(act_ext[time_col], act_ext[target_col], label="Actual", color="#1f77b4")
            elif 'close' in act_ext.columns:
                ax.plot(act_ext[time_col], act_ext['close'], label="Actual", color="#1f77b4")
            y_time = fcst_df[time_col] if time_col in fcst_df.columns else pd.to_datetime(fcst_df.index)
            if 'close' in fcst_df.columns:
                ax.plot(y_time, fcst_df['close'], linestyle="-", color="#ff7f0e", label="Forecast (close)")
            ax.legend(loc="best")
            ax.set_title("Kronos Forecast")
            ax.set_xlabel(time_col)
            ax.set_ylabel(target_col or 'close')
        self.figure = fig
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        try:
            if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                series = []
                if dfx is not None and close_col is not None:
                    series.append((pd.Series(dfx[time_col]), pd.Series(dfx[close_col]), "Actual"))
                y2 = fcst_df['close'] if 'close' in fcst_df.columns else None
                if y_time is not None and y2 is not None:
                    series.append((pd.Series(y_time), pd.Series(y2), "Forecast"))
                if series:
                    self._setup_hover(ax_pe, series)
            else:
                x1 = act_ext[time_col]
                y1 = act_ext[target_col] if target_col in act_ext.columns else act_ext.get('close')
                x2 = y_time
                y2 = fcst_df['close'] if 'close' in fcst_df.columns else None
                series = []
                if x1 is not None and y1 is not None:
                    series.append((x1, y1, "Actual"))
                if x2 is not None and y2 is not None:
                    series.append((pd.Series(x2), pd.Series(y2), "Forecast"))
                if series:
                    self._setup_hover(ax, series)
        except Exception:
            pass
        self.last_act_df = act_ext
        self.last_act_time_col = time_col
        self.last_act_target = target_col if target_col in act_ext.columns else 'close'
        try:
            if bool(self.show_full_moon_var.get()) or bool(self.show_new_moon_var.get()) or bool(self.show_first_quarter_moon_var.get()) or bool(self.show_last_quarter_moon_var.get()):
                self._plot_moon_markers(
                    ax,
                    act_ext,
                    time_col,
                    target_col if target_col in act_ext.columns else 'close',
                    bool(self.show_full_moon_var.get()),
                    bool(self.show_new_moon_var.get()),
                    bool(self.show_first_quarter_moon_var.get()),
                    bool(self.show_last_quarter_moon_var.get()),
                )
        except Exception:
            pass
        try:
            if hasattr(self, 'show_earnings_var') and bool(self.show_earnings_var.get()):
                try:
                    print("Debug: plotting earnings overlay on forecast view", {"symbol": self.symbol_var.get().strip(), "rows": len(act_ext)})
                except Exception:
                    pass
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_earnings_markers(ax_pe, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_earnings_markers(ax, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_insider_var') and bool(self.show_insider_var.get()):
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_insider_sales(ax_pe, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_insider_sales(ax, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_insider_proposed_var') and bool(self.show_insider_proposed_var.get()):
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_insider_proposed_sales(ax_pe, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
                else:
                    self._plot_insider_proposed_sales(ax, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'), self.symbol_var.get().strip())
        except Exception:
            pass
        try:
            if hasattr(self, 'show_volume_dot_var') and bool(self.show_volume_dot_var.get()):
                try:
                    print("Debug: forecast volume dot overlay", {"symbol": self.symbol_var.get().strip(), "rows": len(act_ext)})
                except Exception:
                    pass
                if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
                    self._plot_volume_dot_overlay(ax_pe, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'))
                else:
                    self._plot_volume_dot_overlay(ax, act_ext, time_col, (target_col if target_col in act_ext.columns else 'close'))
        except Exception:
            pass
        if hasattr(self, 'show_pe_eps_var') and bool(self.show_pe_eps_var.get()):
            self._render_markers(ax_pe)
        else:
            self._render_markers(ax)

    def _setup_hover(self, ax, series_list):
        import numpy as np
        from matplotlib.dates import date2num
        # Prepare numeric x arrays per series
        xnums = []
        for x, y, label in series_list:
            try:
                xnums.append(date2num(pd.to_datetime(x)))
            except Exception:
                xnums.append(np.asarray(x, dtype=float))
        fig = ax.figure
        vline = ax.axvline(x=np.nan, color="#999999", linewidth=0.8)
        annot = ax.annotate("", xy=(0, 1), xycoords="axes fraction", xytext=(8, -8), textcoords="offset points", ha="left", va="top")
        annot.set_visible(False)
        def on_move(event):
            if event.inaxes != ax or event.xdata is None:
                annot.set_visible(False)
                fig.canvas.draw_idle()
                return
            ex = float(event.xdata)
            try:
                vline.set_xdata([ex, ex])
            except Exception:
                pass
            lines = []
            for (x, y, label), xs in zip(series_list, xnums):
                if len(y) == len(xs) and len(xs) > 0:
                    try:
                        idx = int(np.clip(np.searchsorted(xs, ex), 0, len(xs) - 1))
                        yi = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
                        xi = x.iloc[idx] if hasattr(x, 'iloc') else x[idx]
                        # Format timestamp when possible
                        try:
                            ts = pd.to_datetime(xi)
                            ts_str = str(ts)
                            self._hover_last_ts = ts
                        except Exception:
                            ts_str = str(xi)
                        lines.append(f"{label} @ {ts_str}: {yi}")
                    except Exception:
                        continue
            annot.set_text("\n".join(lines))
            annot.set_visible(bool(lines))
            fig.canvas.draw_idle()
        try:
            if hasattr(self, '_hover_cid') and self._hover_cid:
                fig.canvas.mpl_disconnect(self._hover_cid)
        except Exception:
            pass
        self._hover_cid = fig.canvas.mpl_connect('motion_notify_event', on_move)
        self._hover_vline = vline
        self._hover_annot = annot
    def _plot_volume_dot_overlay(self, ax, df, time_col, target_col):
        try:
            print("Debug: _plot_volume_dot_overlay called", {
                "enabled": bool(self.show_volume_dot_var.get()) if hasattr(self, "show_volume_dot_var") else None,
                "time_col": time_col,
                "target_col": target_col,
                "df_rows": 0 if df is None else len(df),
                "cols": [] if df is None else list(df.columns),
            })
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass
        except Exception:
            pass
        if ax is None or df is None or df.empty or not time_col:
            return
        dfx = df.copy()
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
        dfx = dfx.dropna(subset=[time_col])
        candidates_close = [target_col, 'close', 'Close', 'Adj Close']
        close_col = next((c for c in candidates_close if c and c in dfx.columns), None)
        candidates_open = ['open', 'Open']
        open_col = next((c for c in candidates_open if c in dfx.columns), None)
        candidates_vol = ['volume', 'Volume', 'vol', 'Vol']
        vol_col = next((c for c in candidates_vol if c in dfx.columns), None)
        try:
            print("Debug: volume dot columns resolved", {"close_col": close_col, "open_col": open_col, "vol_col": vol_col})
        except Exception:
            pass
        if close_col is None or open_col is None or vol_col is None:
            try:
                self.status_var.set("Volume dot: required columns missing (need open/close/volume)")
            except Exception:
                pass
            return
        dfx[close_col] = pd.to_numeric(dfx[close_col], errors="coerce")
        dfx[open_col] = pd.to_numeric(dfx[open_col], errors="coerce")
        dfx[vol_col] = pd.to_numeric(dfx[vol_col], errors="coerce")
        dfx = dfx.dropna(subset=[close_col, open_col, vol_col])
        try:
            print("Debug: volume dot row counts", {"rows_after_dropna": len(dfx)})
        except Exception:
            pass
        x = dfx[time_col]
        y = dfx[close_col]
        try:
            ax.plot(x, y, color="midnightblue", linewidth=1.3, alpha=0.85, label="Close trend")
        except Exception:
            pass
        try:
            colors = ['green' if float(c) > float(o) else 'red' for c, o in zip(dfx[close_col].tolist(), dfx[open_col].tolist())]
            try:
                gc = sum(1 for c in colors if c == 'green')
                rc = sum(1 for c in colors if c == 'red')
                print("Debug: volume dot colors", {"green": gc, "red": rc})
            except Exception:
                pass
        except Exception:
            colors = ['green'] * len(dfx)
        try:
            vols = []
            for vv in dfx[vol_col].tolist():
                try:
                    vols.append(float(vv))
                except Exception:
                    vols.append(float("nan"))
            vols_clean = [v for v in vols if v == v]
            vmin = min(vols_clean) if vols_clean else 0.0
            vmax = max(vols_clean) if vols_clean else 0.0
            try:
                print("Debug: volume dot sizes range", {"vmin": vmin, "vmax": vmax})
            except Exception:
                pass
            sizes = []
            lo, hi = 10.0, 80.0
            if vmin == vmax:
                sizes = [lo] * len(dfx)
            else:
                rng = (vmax - vmin)
                for v in vols:
                    if v != v:
                        sizes.append(lo)
                    else:
                        sizes.append(lo + (v - vmin) / rng * (hi - lo))
            try:
                print("Debug: volume dot sizes sample", {"count": int(len(sizes)), "min": float(min(sizes) if sizes else 0.0), "max": float(max(sizes) if sizes else 0.0)})
            except Exception:
                pass
        except Exception:
            sizes = [10.0] * len(dfx)
        try:
            ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5, label="Volume dots")
            try:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="best")
            except Exception:
                pass
            try:
                self.status_var.set(f"Volume dot: plotted {len(dfx)} points")
                print("Debug: volume dot plotted", {"points": len(dfx)})
            except Exception:
                pass
            try:
                if hasattr(self, 'canvas') and self.canvas:
                    self.canvas.draw()
                else:
                    ax.figure.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    def show_marker_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("Add Marker")
        frm = ttk.Frame(win)
        frm.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        ttk.Label(frm, text="Date/Time (optional)").grid(row=0, column=0, sticky=tk.W)
        dt_var = tk.StringVar()
        dt_entry = ttk.Entry(frm, textvariable=dt_var, width=24)
        dt_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(frm, text="Text").grid(row=1, column=0, sticky=tk.W)
        txt_var = tk.StringVar()
        txt_entry = ttk.Entry(frm, textvariable=txt_var, width=32)
        txt_entry.grid(row=1, column=1, sticky=tk.W)
        ttk.Label(frm, text="Series").grid(row=2, column=0, sticky=tk.W)
        series_var = tk.StringVar(value="Actual")
        series_cb = ttk.Combobox(frm, textvariable=series_var, values=["Actual","Forecast"], width=12)
        series_cb.grid(row=2, column=1, sticky=tk.W)
        def start_click():
            # Update persistent marker inputs, then enable click mode
            self.marker_series_var.set(series_var.get())
            self.marker_text_var.set(txt_var.get().strip())
            self._toggle_marker_mode(force_on=True)
            try:
                win.destroy()
            except Exception:
                pass
        ttk.Button(frm, text="Click On Chart To Place", command=start_click).grid(row=3, column=0, columnspan=2, pady=6)

    def place_marker(self, dt_text, text, series, ts_override=None):
        try:
            ts = ts_override if ts_override is not None else (pd.to_datetime(dt_text, errors="coerce") if dt_text else getattr(self, '_hover_last_ts', None))
        except Exception:
            self.status_var.set("Invalid date/time for marker")
            return
        ax = self.figure.gca() if self.figure else None
        if ax is None:
            self.status_var.set("No plot available")
            return
        if ts is None:
            self.status_var.set("No cursor position or date provided")
            return
        if series == "Forecast" and self.last_fcst_df is not None and self.last_time_col:
            idx = pd.DatetimeIndex(pd.to_datetime(self.last_fcst_df.index if self.last_time_col not in self.last_fcst_df.columns else self.last_fcst_df[self.last_time_col]).dropna().sort_values())
            i = idx.get_indexer([ts], method='pad')
            if len(i) and i[0] >= 0:
                ts_adj = idx[i[0]]
                try:
                    y = self.last_fcst_df.loc[ts_adj][self.last_target_col] if self.last_target_col in self.last_fcst_df.columns else self.last_fcst_df.loc[ts_adj]['close']
                except Exception:
                    y = None
            else:
                ts_adj = None
                y = None
        else:
            if self.last_act_df is None or self.last_act_time_col is None:
                self.status_var.set("No actual series available")
                return
            tcol = self.last_act_time_col
            idx = pd.DatetimeIndex(pd.to_datetime(self.last_act_df[tcol]).dropna().sort_values())
            i = idx.get_indexer([ts], method='pad')
            if len(i) and i[0] >= 0:
                ts_adj = idx[i[0]]
                try:
                    y = self.last_act_df.loc[self.last_act_df[tcol] == ts_adj][self.last_act_target].iloc[0]
                except Exception:
                    y = None
            else:
                ts_adj = None
                y = None
        if ts_adj is None or y is None:
            self.status_var.set("Marker not placed: no matching timestamp")
            return
        ax.scatter([ts_adj], [y], s=40, color="#2ca02c", marker="o")
        if text:
            ax.annotate(text, (ts_adj, y), xytext=(8, -8), textcoords="offset points")
        if bool(self.show_marker_vlines_var.get()):
            try:
                ax.axvline(x=ts_adj, color="#2ca02c", alpha=0.3)
            except Exception:
                pass
        self.canvas.draw_idle()
        try:
            self.markers.append((ts_adj, y, text, series))
        except Exception:
            pass

    def _render_markers(self, ax):
        if ax is None or not self.markers:
            return
        if bool(self.show_marker_vlines_var.get()):
            for ts_adj, y, text, series in self.markers:
                try:
                    ax.axvline(x=ts_adj, color="#2ca02c", alpha=0.3)
                except Exception:
                    continue

    def _toggle_marker_mode(self, force_on=False):
        on = force_on or bool(self.marker_mode_var.get())
        if on:
            self._enable_marker_click()
        else:
            self._disable_marker_click()

    def _enable_marker_click(self):
        fig = self.figure
        if fig is None:
            self.status_var.set("No plot available")
            return
        def on_click(event):
            if event.inaxes is None or event.xdata is None:
                return
            try:
                from matplotlib.dates import num2date
                dt = num2date(event.xdata)
                if getattr(dt, 'tzinfo', None) is not None:
                    dt = dt.replace(tzinfo=None)
                ts = pd.to_datetime(dt)
            except Exception:
                return
            try:
                self.place_marker("", self.marker_text_var.get().strip(), self.marker_series_var.get(), ts_override=ts)
            except Exception:
                pass
        try:
            if self._marker_click_cid:
                fig.canvas.mpl_disconnect(self._marker_click_cid)
        except Exception:
            pass
        self._marker_click_cid = fig.canvas.mpl_connect('button_press_event', on_click)
        self.status_var.set("Marker mode: click on chart to add markers")

    def _disable_marker_click(self):
        fig = self.figure
        if fig is None:
            return
        try:
            if self._marker_click_cid:
                fig.canvas.mpl_disconnect(self._marker_click_cid)
        except Exception:
            pass
        self._marker_click_cid = None
        self.status_var.set("Marker mode off")

    def _compute_moon_dates(self, start_ts, end_ts):
        import ephem
        import datetime as _dt
        fulls = []
        news = []
        firsts = []
        lasts = []
        cur = start_ts.to_pydatetime() if hasattr(start_ts, 'to_pydatetime') else _dt.datetime.fromtimestamp(pd.to_datetime(start_ts).timestamp())
        end_py = end_ts.to_pydatetime() if hasattr(end_ts, 'to_pydatetime') else _dt.datetime.fromtimestamp(pd.to_datetime(end_ts).timestamp())
        f = ephem.next_full_moon(cur)
        while f.datetime() <= end_py:
            fulls.append(pd.to_datetime(f.datetime()))
            f = ephem.next_full_moon(f)
        n = ephem.next_new_moon(cur)
        while n.datetime() <= end_py:
            news.append(pd.to_datetime(n.datetime()))
            n = ephem.next_new_moon(n)
        try:
            q1 = ephem.next_first_quarter_moon(cur)
            while q1.datetime() <= end_py:
                firsts.append(pd.to_datetime(q1.datetime()))
                q1 = ephem.next_first_quarter_moon(q1)
        except Exception:
            pass
        try:
            q3 = ephem.next_last_quarter_moon(cur)
            while q3.datetime() <= end_py:
                lasts.append(pd.to_datetime(q3.datetime()))
                q3 = ephem.next_last_quarter_moon(q3)
        except Exception:
            pass
        return fulls, news, firsts, lasts

    def _plot_moon_markers(self, ax, df, time_col, target_col, show_full, show_new, show_first, show_last):
        if df is None or df.empty:
            return
        idx = pd.DatetimeIndex(pd.to_datetime(df[time_col]).dropna().sort_values())
        start = idx.min()
        end = idx.max()
        fulls, news, firsts, lasts = self._compute_moon_dates(start, end)
        def adjust(dt):
            ts = pd.to_datetime(dt)
            i = idx.get_indexer([ts], method='pad')
            if len(i) and i[0] >= 0:
                return idx[i[0]]
            return None
        xs_f = []
        ys_f = []
        xs_n = []
        ys_n = []
        xs_q1 = []
        ys_q1 = []
        xs_q3 = []
        ys_q3 = []
        # Build a mapping Series for fast y lookup
        df_map = df.copy()
        df_map[time_col] = pd.to_datetime(df_map[time_col])
        df_map = df_map.dropna(subset=[time_col, target_col])
        for d in fulls:
            a = adjust(d)
            if a is not None:
                xs_f.append(a)
                try:
                    ys_f.append(df_map.loc[df_map[time_col] == a, target_col].iloc[0])
                except Exception:
                    continue
        for d in news:
            a = adjust(d)
            if a is not None:
                xs_n.append(a)
                try:
                    ys_n.append(df_map.loc[df_map[time_col] == a, target_col].iloc[0])
                except Exception:
                    continue
        for d in firsts:
            a = adjust(d)
            if a is not None:
                xs_q1.append(a)
                try:
                    ys_q1.append(df_map.loc[df_map[time_col] == a, target_col].iloc[0])
                except Exception:
                    continue
        for d in lasts:
            a = adjust(d)
            if a is not None:
                xs_q3.append(a)
                try:
                    ys_q3.append(df_map.loc[df_map[time_col] == a, target_col].iloc[0])
                except Exception:
                    continue
        if show_full and xs_f:
            ax.scatter(xs_f, ys_f, s=30, color="#d62728", label="Full Moon")
        if show_new and xs_n:
            ax.scatter(xs_n, ys_n, s=30, color="#9467bd", label="New Moon")
        if show_first and xs_q1:
            ax.scatter(xs_q1, ys_q1, s=30, color="#ff69b4", marker="^", label="1st Quarter")
        if show_last and xs_q3:
            ax.scatter(xs_q3, ys_q3, s=30, color="#ffd700", marker="s", label="Last Quarter")
        try:
            import ephem
            import datetime as _dt
            now_py = _dt.datetime.utcnow()
            end_py = end.to_pydatetime() if hasattr(end, "to_pydatetime") else _dt.datetime.fromtimestamp(pd.to_datetime(end).timestamp())
            labels_done = set()
            future_events = []
            if show_full:
                dt = ephem.next_full_moon(now_py)
                while dt.datetime() <= end_py:
                    a = adjust(pd.to_datetime(dt.datetime()))
                    if a is not None:
                        lbl = "Next Full Moon" if "full" not in labels_done else None
                        ax.axvline(x=a, color="#d62728", alpha=0.35, linestyle="--", label=lbl)
                        labels_done.add("full")
                    dt = ephem.next_full_moon(dt)
                if dt.datetime() > end_py:
                    future_events.append(("Next Full Moon", pd.to_datetime(dt.datetime()), "#d62728", "full"))
            if show_new:
                dt = ephem.next_new_moon(now_py)
                while dt.datetime() <= end_py:
                    a = adjust(pd.to_datetime(dt.datetime()))
                    if a is not None:
                        lbl = "Next New Moon" if "new" not in labels_done else None
                        ax.axvline(x=a, color="#9467bd", alpha=0.35, linestyle="--", label=lbl)
                        labels_done.add("new")
                    dt = ephem.next_new_moon(dt)
                if dt.datetime() > end_py:
                    future_events.append(("Next New Moon", pd.to_datetime(dt.datetime()), "#9467bd", "new"))
            if show_first:
                dt = ephem.next_first_quarter_moon(now_py)
                while dt.datetime() <= end_py:
                    a = adjust(pd.to_datetime(dt.datetime()))
                    if a is not None:
                        lbl = "Next 1st Quarter" if "q1" not in labels_done else None
                        ax.axvline(x=a, color="#ff69b4", alpha=0.35, linestyle="--", label=lbl)
                        labels_done.add("q1")
                    dt = ephem.next_first_quarter_moon(dt)
                if dt.datetime() > end_py:
                    future_events.append(("Next 1st Quarter", pd.to_datetime(dt.datetime()), "#ff69b4", "q1"))
            if show_last:
                dt = ephem.next_last_quarter_moon(now_py)
                while dt.datetime() <= end_py:
                    a = adjust(pd.to_datetime(dt.datetime()))
                    if a is not None:
                        lbl = "Next Last Quarter" if "q3" not in labels_done else None
                        ax.axvline(x=a, color="#ffd700", alpha=0.35, linestyle="--", label=lbl)
                        labels_done.add("q3")
                    dt = ephem.next_last_quarter_moon(dt)
                if dt.datetime() > end_py:
                    future_events.append(("Next Last Quarter", pd.to_datetime(dt.datetime()), "#ffd700", "q3"))
            if future_events:
                try:
                    new_right = max(ev[1] for ev in future_events)
                    ax.set_xlim(right=new_right)
                except Exception:
                    pass
                for lbl, xline, col, key in future_events:
                    try:
                        l = lbl if key not in labels_done else None
                        ax.axvline(x=xline, color=col, alpha=0.35, linestyle="--", label=l)
                        labels_done.add(key)
                    except Exception:
                        continue
        except Exception:
            pass
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="best")

    def update_plot(self):
        if self.last_fcst_df is None:
            messagebox.showinfo("Info", "Run a forecast first")
            return
        self.plot_results(self.last_fcst_df, self.last_id_col, self.last_time_col, self.last_target_col, self.last_levels, self.last_quantiles)

    def show_table(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Load a TSV file first")
            return
        try:
            if self.table_win and self.table_win.winfo_exists():
                self.table_win.lift()
                return
        except Exception:
            pass
        win = tk.Toplevel(self.root)
        win.title("Data Table")
        self.table_win = win
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)
        df = self.df.copy()
        id_col = self.id_col_var.get() if self.id_col_var.get() else None
        if id_col and id_col in df.columns and self.series_id_var.get():
            df = df[df[id_col] == self.series_id_var.get()]
        cols = list(df.columns)
        tree = ttk.Treeview(frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, stretch=True)
        limit = min(len(df), 1000)
        for _, row in df.head(limit).iterrows():
            values = [str(row.get(c)) for c in cols]
            tree.insert("", "end", values=values)
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

    def _plot_earnings_markers(self, ax, df, time_col, target_col, symbol):
        try:
            print("Debug: _plot_earnings_markers called", {"symbol": symbol, "time_col": time_col, "target_col": target_col, "df_rows": 0 if df is None else len(df)})
        except Exception:
            pass
        if not symbol or df is None or df.empty:
            try:
                print("Debug: earnings early-exit", {"has_symbol": bool(symbol), "df_empty": (df is None or df.empty)})
            except Exception:
                pass
            return
        try:
            import yfinance as yf
            import pandas as pd
        except Exception:
            return
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(df[time_col]).dropna().sort_values())
            start = idx.min()
            end = idx.max()
        except Exception:
            return
        try:
            print("Debug: earnings range", {"start": str(start), "end": str(end), "index_len": len(idx)})
        except Exception:
            pass
        dates = pd.Series([], dtype='datetime64[ns]')
        surprise_map = {}
        try:
            tk = yf.Ticker(symbol.strip().upper())
            cal = tk.get_earnings_dates(limit=100)
            if cal is not None and len(cal) > 0:
                if hasattr(cal, 'columns') and 'Earnings Date' in list(cal.columns):
                    dates = pd.to_datetime(cal['Earnings Date'], utc=True).tz_convert(None)
                elif hasattr(cal, 'index'):
                    dates = pd.to_datetime(cal.index, utc=True).tz_convert(None)
            if not dates.empty:
                try:
                    print("Earnings (yfinance):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                except Exception:
                    pass
            if dates.empty:
                try:
                    cal2 = tk.calendar
                except Exception:
                    cal2 = None
                parsed_cal = []
                try:
                    if cal2:
                        if isinstance(cal2, dict):
                            ed = cal2.get('Earnings Date')
                            if ed:
                                if isinstance(ed, (list, tuple)):
                                    for d in ed:
                                        try:
                                            parsed_cal.append(pd.to_datetime(d, utc=True))
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        parsed_cal.append(pd.to_datetime(ed, utc=True))
                                    except Exception:
                                        pass
                        else:
                            try:
                                dfc = pd.DataFrame(cal2).T
                                if 'Earnings Date' in dfc.columns:
                                    vals = dfc['Earnings Date'].values.tolist()
                                    for d in vals:
                                        try:
                                            parsed_cal.append(pd.to_datetime(d, utc=True))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                except Exception:
                    parsed_cal = []
                if parsed_cal:
                    try:
                        dates = pd.to_datetime(parsed_cal, utc=True).tz_convert(None)
                        print("Earnings (yfinance calendar):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                    except Exception:
                        pass
            try:
                edf = tk.earnings_dates
            except Exception:
                edf = None
            if edf is not None and hasattr(edf, 'index') and len(edf) > 0:
                try:
                    idx_dates = pd.to_datetime(edf.index, utc=True).tz_convert(None)
                    try:
                        print("Earnings (yfinance table):", [d.strftime("%Y-%m-%d") for d in idx_dates.tolist()])
                    except Exception:
                        pass
                    try:
                        if 'Surprise(%)' in edf.columns:
                            s = pd.to_numeric(edf['Surprise(%)'], errors='coerce')
                            for ts, val in zip(idx_dates.tolist(), s.tolist()):
                                if pd.notna(val):
                                    surprise_map[pd.to_datetime(ts)] = float(val)
                            print("Debug: earnings surprises from yfinance table", {"count": len(surprise_map)})
                        # Fallback: compute Surprise(%) if missing using Reported and Estimate
                        if len(surprise_map) == 0:
                            est_col = None
                            rep_col = None
                            for c in ['EPS Estimate','Estimate','epsEstimate','estimatedEPS']:
                                if c in edf.columns:
                                    est_col = c
                                    break
                            for c in ['Reported EPS','Reported','reportedEPS','reported']:
                                if c in edf.columns:
                                    rep_col = c
                                    break
                            if est_col and rep_col:
                                est = pd.to_numeric(edf[est_col], errors='coerce')
                                rep = pd.to_numeric(edf[rep_col], errors='coerce')
                                surp = ((rep - est) / est * 100.0)
                                for ts, val in zip(idx_dates.tolist(), surp.tolist()):
                                    if pd.notna(val):
                                        surprise_map[pd.to_datetime(ts)] = float(val)
                                print("Debug: earnings surprises computed from table", {"count": len(surprise_map)})
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        if dates.empty:
            try:
                import asyncio
                import json
                import urllib.request
                req = urllib.request.Request(
                    url="http://127.0.0.1:8088/tool/get_earnings_dates",
                    data=json.dumps({"symbol": symbol.strip().upper(), "limit": 100}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read()
                    obj = json.loads(body.decode("utf-8"))
                    data = obj.get("data", [])
                    parsed_local = []
                    parsed_surp = []
                    for item in data:
                        d = item.get("date") if isinstance(item, dict) else item
                        if d:
                            try:
                                parsed_local.append(pd.to_datetime(d, utc=True))
                            except Exception:
                                pass
                        if isinstance(item, dict):
                            for k in ["Surprise(%)", "surprise", "Surprise"]:
                                if k in item and item[k] is not None:
                                    try:
                                        parsed_surp.append((pd.to_datetime(d, utc=True), float(item[k])))
                                        break
                                    except Exception:
                                        pass
                            # Fallback compute from estimate/reported if surprise not explicitly provided
                            if not any(k in item for k in ["Surprise(%)", "surprise", "Surprise"]):
                                # Try multiple key aliases
                                est = None
                                rep = None
                                for kc in ["EPS Estimate","Estimate","epsEstimate","estimatedEPS"]:
                                    if kc in item and item[kc] is not None:
                                        try:
                                            est = float(item[kc])
                                            break
                                        except Exception:
                                            pass
                                for kc in ["Reported EPS","Reported","reportedEPS","reported"]:
                                    if kc in item and item[kc] is not None:
                                        try:
                                            rep = float(item[kc])
                                            break
                                        except Exception:
                                            pass
                                try:
                                    if est is not None and rep is not None and est != 0:
                                        val = (rep - est) / est * 100.0
                                        parsed_surp.append((pd.to_datetime(d, utc=True), val))
                                except Exception:
                                    pass
                    if parsed_local:
                        dates = pd.to_datetime(parsed_local, utc=True).tz_convert(None)
                        try:
                            print("Earnings (MCP HTTP):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                        except Exception:
                            pass
                    if parsed_surp:
                        try:
                            for ts, val in parsed_surp:
                                surprise_map[pd.to_datetime(ts).tz_convert(None)] = val
                            print("Debug: earnings surprises from MCP HTTP", {"count": len(parsed_surp)})
                        except Exception:
                            pass
            except Exception:
                pass
        if dates.empty:
            try:
                import os
                import sys
                import asyncio
                try:
                    from spoonos_stock_agent import MCPClient
                except ImportError:
                    candidates = [
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spoon-core-jue")),
                        r"C:\\Users\\juesh\\jules\\spoon-core-jue",
                        os.path.expanduser(r"~\\jules\\spoon-core-jue"),
                    ]
                    for p in candidates:
                        if os.path.isdir(p) and p not in sys.path:
                            sys.path.append(p)
                    from spoonos_stock_agent import MCPClient
                async def fetch():
                    client = MCPClient(server_name="stock-mcp")
                    tools = [
                        "get_earnings_dates",
                        "get_stock_earnings_calendar",
                        "get_earnings_calendar",
                        "get_company_events",
                    ]
                    for t in tools:
                        try:
                            res = await client.call_tool(t, {"symbol": symbol.strip().upper()})
                            if isinstance(res, dict):
                                data = res.get("data")
                                if data:
                                    return data
                        except Exception:
                            continue
                    return []
                raw = asyncio.run(fetch())
                parsed = []
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            for k in ["date", "earningsDate", "reportedDate", "datetime", "time"]:
                                if k in item and item[k]:
                                    try:
                                        parsed.append(pd.to_datetime(item[k], utc=True))
                                        break
                                    except Exception:
                                        pass
                        else:
                            try:
                                parsed.append(pd.to_datetime(item, utc=True))
                            except Exception:
                                pass
                if parsed:
                    dates = pd.to_datetime(parsed, utc=True).tz_convert(None)
                    try:
                        print("Earnings (MCP client):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                    except Exception:
                        pass
                try:
                    self.status_var.set(f"Earnings (MCP): {len(dates)} dates fetched")
                except Exception:
                    pass
            except Exception:
                pass
        total = len(dates)
        try:
            print("Debug: earnings fetched total", {"count": total})
        except Exception:
            pass
        dates_full = dates.copy()
        dates = dates[(dates >= start) & (dates <= end)]
        try:
            print("Debug: earnings in-range", {"count": len(dates)})
        except Exception:
            pass
        if dates.empty:
            try:
                self.status_var.set(f"Earnings: {total} fetched, 0 in current range")
            except Exception:
                pass
            return
        def adjust(dt):
            i = idx.get_indexer([pd.to_datetime(dt)], method='pad')
            if len(i) and i[0] >= 0:
                return idx[i[0]]
            return None
        df_map = df.copy()
        df_map[time_col] = pd.to_datetime(df_map[time_col])
        df_map = df_map.dropna(subset=[time_col, target_col]) if target_col in df_map.columns else df_map.dropna(subset=[time_col])
        try:
            print("Debug: earnings df_map", {"rows": len(df_map), "has_target": target_col in df_map.columns})
        except Exception:
            pass
        xs = []
        ys = []
        mapped = 0
        unmapped = 0
        surprises = []
        for d in dates:
            a = adjust(d)
            if a is not None:
                try:
                    yv = df_map.loc[df_map[time_col] == a, target_col].iloc[0]
                except Exception:
                    yv = None
                xs.append(a)
                ys.append(yv)
                mapped += 1
                try:
                    dn = pd.to_datetime(d)
                    if getattr(dn, 'tzinfo', None) is not None:
                        dn = dn.tz_convert(None)
                    sv = surprise_map.get(pd.to_datetime(dn))
                except Exception:
                    sv = None
                surprises.append(sv)
            else:
                unmapped += 1
        try:
            print("Debug: earnings mapping", {"mapped": mapped, "unmapped": unmapped, "xs": len(xs), "ys": len(ys)})
        except Exception:
            pass
        if xs:
            try:
                pts_x = [x for x, y in zip(xs, ys) if y is not None]
                pts_y = [y for y in ys if y is not None]
                try:
                    print("Debug: earnings scatter points", {"scatter_count": len(pts_x)})
                except Exception:
                    pass
                if pts_x:
                    ax.scatter(pts_x, pts_y, s=30, color="#ff9900", label="Earnings")
                    try:
                        ann = [(x, y, s) for x, y, s in zip(xs, ys, surprises) if y is not None and s is not None]
                        for x, y, s in ann:
                            offs = 10 if s >= 0 else -12
                            ax.annotate(f"{s:.2f}%", (x, y), xytext=(0, offs), textcoords="offset points", color=("#2ca02c" if s >= 0 else "#d62728"))
                        print("Debug: earnings surprises annotated", {"count": len(ann)})
                    except Exception:
                        pass
                if bool(self.show_marker_vlines_var.get()):
                    try:
                        print("Debug: earnings vlines", {"vline_count": len(xs)})
                    except Exception:
                        pass
                    for x in xs:
                        ax.axvline(x=x, color="#ff0000", alpha=0.8)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="best")
                try:
                    self.status_var.set(f"Earnings: plotted {len(xs)} markers (scatter {len(pts_x)})")
                except Exception:
                    pass
            except Exception:
                pass
        try:
            upcoming = dates_full[dates_full > end]
            if len(upcoming) > 0:
                next_dt = pd.to_datetime(upcoming.min())
                try:
                    ax.axvline(x=next_dt, color="#ff0000", alpha=0.8, linestyle="--")
                except Exception:
                    pass
                try:
                    from matplotlib.dates import date2num
                    l, r = ax.get_xlim()
                    nx = date2num(next_dt)
                    if nx > r:
                        ax.set_xlim(l, nx)
                    print("Debug: upcoming earnings vline", {"date": str(next_dt)})
                except Exception:
                    pass
        except Exception:
            pass

    def _fetch_insider_sales(self, symbol):
        try:
            import yfinance as yf
            import pandas as pd
            import requests
            from io import StringIO
        except Exception:
            return None
        df = None
        try:
            ticker = yf.Ticker(symbol.strip().upper())
            df = ticker.insider_transactions
        except Exception:
            df = None
        if df is None or len(df) == 0:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                url = f"https://finviz.com/quote.ashx?t={symbol.strip().upper()}&p=d"
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                tables = pd.read_html(StringIO(resp.text))
                target = None
                for table in tables:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.get_level_values(-1)
                    if 'Insider Trading' in table.columns:
                        target = table
                        break
                if target is None:
                    return None
                rename_map = {
                    'Insider Trading': 'Who',
                    'Relationship': 'Relationship',
                    'Date': 'Date',
                    'Transaction': 'TransactionText',
                    'Cost': 'Cost',
                    '#Shares': 'Shares',
                    '# Shares': 'Shares',
                    'Value ($)': 'Value'
                }
                df = target.rename(columns={k: v for k, v in rename_map.items() if k in target.columns})
                required_cols = {'Date', 'Who', 'TransactionText', 'Shares', 'Value', 'Cost'}
                if not required_cols.issubset(df.columns):
                    return None
                df['TransactionText'] = df['TransactionText'].astype(str)
                df = df[df['TransactionText'].str.contains("Sale", case=False, na=False)].copy()
                if df.empty:
                    return None
                df['Date'] = pd.to_datetime(df['Date'], format="%b %d '%y", errors='coerce')
                df['Shares'] = pd.to_numeric(df['Shares'].astype(str).str.replace(',', ''), errors='coerce')
                df['Value'] = pd.to_numeric(df['Value'].astype(str).str.replace(',', ''), errors='coerce')
                df['Cost'] = pd.to_numeric(df['Cost'].astype(str).str.replace(',', ''), errors='coerce')
                df.dropna(subset=['Date', 'Shares', 'Value', 'Cost'], inplace=True)
            except Exception:
                return None
            return df[['Date', 'Who', 'TransactionText', 'Cost', 'Shares', 'Value']].sort_values('Date')
        try:
            df = df.reset_index()
            rename_map = {
                'Start Date': 'Date',
                'Insider': 'Who',
                'Transaction': 'TransactionType',
                'Text': 'TransactionText',
                'Value': 'TotalValue',
                '#Shares': 'Shares'
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            cols = [c for c in ('TransactionType', 'TransactionText', 'Transaction') if c in df.columns]
            if not cols:
                return None
            sale_mask = pd.Series(False, index=df.index)
            for c in cols:
                sale_mask |= df[c].astype(str).str.contains("Sale", case=False, na=False)
            sdf = df[sale_mask].copy()
            if sdf.empty:
                return None
            sdf['Date'] = pd.to_datetime(sdf['Date'])
            sdf['Shares'] = pd.to_numeric(sdf['Shares'], errors='coerce')
            if 'TotalValue' in sdf.columns:
                sdf['TotalValue'] = pd.to_numeric(sdf['TotalValue'], errors='coerce')
            sdf = sdf[sdf['Shares'] > 0]
            if 'TotalValue' in sdf.columns:
                sdf['Cost'] = sdf['TotalValue'] / sdf['Shares']
                sdf.rename(columns={'TotalValue': 'Value'}, inplace=True)
            else:
                if 'Cost' in sdf.columns:
                    sdf['Value'] = sdf['Cost'] * sdf['Shares']
            sdf = sdf.dropna(subset=['Date', 'Shares'])
            sdf = sdf.sort_values('Date')
            return sdf[['Date', 'Who', 'TransactionText' if 'TransactionText' in sdf.columns else 'Transaction', 'Cost', 'Shares', 'Value']].rename(columns={'TransactionText': 'Transaction'})
        except Exception:
            return None

    def _plot_insider_sales(self, ax, df, time_col, target_col, symbol):
        print("Debug: _plot_insider_sales called", {"symbol": symbol, "time_col": time_col, "target_col": target_col, "df_rows": 0 if df is None else len(df)})
        if df is None or df.empty:
            return
        if not symbol:
            try:
                if 'unique_id' in df.columns:
                    vals = [str(v).strip() for v in df['unique_id'].dropna().tolist() if str(v).strip()]
                    symbol = vals[0] if vals else symbol
            except Exception:
                pass
        symbol = (symbol or "").strip().upper()
        if not symbol:
            try:
                self.status_var.set("Insider: no symbol set; skip")
            except Exception:
                pass
            return
        try:
            print("Debug: fetching insider sales for", symbol)
            sales = self._fetch_insider_sales(symbol)
            try:
                print("Debug: fetched insider rows", 0 if sales is None else len(sales))
            except Exception:
                pass
        except Exception as e:
            try:
                self.status_var.set(f"Insider: error fetching for {symbol}: {e}")
            except Exception:
                pass
            sales = None
        if sales is None or sales.empty:
            try:
                self.status_var.set(f"Insider: no sales for {symbol}")
            except Exception:
                pass
            return
        total_count = len(sales)
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.dropna(subset=[time_col])
        df_sorted = df_sorted.sort_values(time_col)
        idx = pd.DatetimeIndex(df_sorted[time_col].values)
        start = idx.min()
        end = idx.max()
        sales_full = sales.copy()
        sales = sales[(sales_full['Date'] >= start) & (sales_full['Date'] <= end)]
        if sales.empty:
            try:
                self.status_var.set(f"Insider: fetched {total_count} total, 0 in range")
            except Exception:
                pass
            return
        def adjust(dt):
            i = idx.get_indexer([pd.to_datetime(dt)], method='pad')
            if len(i) and i[0] >= 0:
                return i[0]
            return None
        xs = []
        ys = []
        sizes = []
        vals = pd.to_numeric(sales.get('Value', pd.Series([0]*len(sales))), errors='coerce').fillna(0.0)
        vmax = float(vals.max()) if len(vals) else 0.0
        mapped = 0
        mapped_records = []
        for _, row in sales.iterrows():
            pos = adjust(row['Date'])
            if pos is not None and pos >= 0 and pos < len(df_sorted):
                ts_adj = idx[pos]
                xs.append(ts_adj)
                try:
                    yv = (pd.to_numeric(df_sorted[target_col], errors='coerce') if target_col in df_sorted.columns else pd.to_numeric(df_sorted['close'], errors='coerce')).iloc[pos]
                except Exception:
                    yv = None
                ys.append(yv)
                if vmax > 0:
                    sizes.append(50 + 400 * (row.get('Value', 0.0) / vmax))
                else:
                    sizes.append(80)
                mapped += 1
                try:
                    rec = {
                        "Who": row.get("Who"),
                        "Transaction": row.get("TransactionText") if "TransactionText" in sales.columns else (row.get("Transaction") if "Transaction" in sales.columns else row.get("TransactionType")),
                        "Date": row.get("Date"),
                        "Shares": row.get("Shares"),
                        "Value": row.get("Value") if "Value" in sales.columns else row.get("TotalValue"),
                        "Cost": row.get("Cost"),
                    }
                    pct = None
                    for kc in ["SharesOwned","Owned","Holdings","Ownership","%Owned","Percent"]:
                        v = row.get(kc)
                        try:
                            if v is not None:
                                if kc in ["Ownership","%Owned","Percent"] and isinstance(v, str):
                                    vs = str(v).strip().strip("%")
                                    pct = float(vs)
                                    break
                                sv = float(row.get("Shares")) if row.get("Shares") is not None else None
                                ov = float(v)
                                if sv is not None and ov and ov != 0:
                                    pct = 100.0 * sv / ov
                                    break
                        except Exception:
                            continue
                    rec["PercentSold"] = pct
                    mapped_records.append(rec)
                except Exception:
                    pass
        pts_x = [x for x, y in zip(xs, ys) if y is not None]
        pts_y = [y for y in ys if y is not None]
        pts_s = [s for s, y in zip(sizes, ys) if y is not None]
        try:
            if pts_x:
                print("Debug: plotting insider scatter points", len(pts_x))
                coll = ax.scatter(pts_x, pts_y, s=pts_s, color="#8c564b", edgecolors="black", alpha=0.8, label="Insider Sales")
                if bool(self.insider_tooltips_var.get()):
                    try:
                        import mplcursors
                        disp_records = mapped_records
                        cur = mplcursors.cursor(coll, hover=True)
                        def _on_add(sel):
                            try:
                                i = int(sel.index)
                            except Exception:
                                return
                            if i < 0 or i >= len(disp_records):
                                return
                            r = disp_records[i]
                            who = str(r.get("Who") or "").strip()
                            tx = str(r.get("Transaction") or "Sale").strip()
                            dt = r.get("Date")
                            sh = r.get("Shares")
                            val = r.get("Value")
                            cost = r.get("Cost")
                            pct = r.get("PercentSold")
                            txt = f"{who} ({tx})"
                            if dt is not None:
                                txt += f"\nDate: {pd.to_datetime(dt)}"
                            if sh is not None:
                                try:
                                    txt += f"\nShares: {int(float(sh)):,}"
                                except Exception:
                                    txt += f"\nShares: {sh}"
                            if val is not None:
                                try:
                                    txt += f"\nValue: ${float(val):,.0f}"
                                except Exception:
                                    txt += f"\nValue: {val}"
                            if cost is not None:
                                try:
                                    txt += f"\nPrice: ${float(cost):.2f}"
                                except Exception:
                                    txt += f"\nPrice: {cost}"
                            if pct is not None:
                                try:
                                    txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                except Exception:
                                    txt += f"\n% Holding Sold: {pct}"
                            sel.annotation.set_text(txt)
                        cur.connect("add", _on_add)
                        try:
                            self.status_var.set("Insider tooltips: hover enabled")
                        except Exception:
                            pass
                    except Exception:
                        try:
                            coll.set_picker(True)
                            disp_records = mapped_records
                            fig = ax.figure
                            annot = ax.annotate("", xy=(0, 0), xytext=(8, 8), textcoords="offset points")
                            annot.set_visible(False)
                            def _on_pick(ev):
                                inds = getattr(ev, "ind", [])
                                if not inds:
                                    return
                                i = int(inds[0])
                                if i < 0 or i >= len(disp_records):
                                    return
                                r = disp_records[i]
                                x = pts_x[i]
                                y = pts_y[i]
                                who = str(r.get("Who") or "").strip()
                                tx = str(r.get("Transaction") or "Sale").strip()
                                dt = r.get("Date")
                                sh = r.get("Shares")
                                val = r.get("Value")
                                cost = r.get("Cost")
                                pct = r.get("PercentSold")
                                txt = f"{who} ({tx})"
                                if dt is not None:
                                    txt += f"\nDate: {pd.to_datetime(dt)}"
                                if sh is not None:
                                    try:
                                        txt += f"\nShares: {int(float(sh)):,}"
                                    except Exception:
                                        txt += f"\nShares: {sh}"
                                if val is not None:
                                    try:
                                        txt += f"\nValue: ${float(val):,.0f}"
                                    except Exception:
                                        txt += f"\nValue: {val}"
                                if cost is not None:
                                    try:
                                        txt += f"\nPrice: ${float(cost):.2f}"
                                    except Exception:
                                        txt += f"\nPrice: {cost}"
                                if pct is not None:
                                    try:
                                        txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                    except Exception:
                                        txt += f"\n% Holding Sold: {pct}"
                                annot.set_text(txt)
                                annot.xy = (x, y)
                                annot.set_visible(True)
                                fig.canvas.draw_idle()
                            fig.canvas.mpl_connect("pick_event", _on_pick)
                            try:
                                self.status_var.set("Insider tooltips: click on bubble for details")
                            except Exception:
                                pass
                        except Exception:
                            pass
                if bool(self.insider_tooltips_var.get()):
                    try:
                        import mplcursors
                        disp_records = mapped_records
                        cur = mplcursors.cursor(coll, hover=True)
                        def _on_add(sel):
                            try:
                                i = int(sel.index)
                            except Exception:
                                return
                            if i < 0 or i >= len(disp_records):
                                return
                            r = disp_records[i]
                            who = str(r.get("Who") or "").strip()
                            tx = str(r.get("Transaction") or "Sale").strip()
                            dt = r.get("Date")
                            sh = r.get("Shares")
                            val = r.get("Value")
                            cost = r.get("Cost")
                            pct = r.get("PercentSold")
                            txt = f"{who} ({tx})"
                            if dt is not None:
                                txt += f"\nDate: {pd.to_datetime(dt)}"
                            if sh is not None:
                                try:
                                    txt += f"\nShares: {int(float(sh)):,}"
                                except Exception:
                                    txt += f"\nShares: {sh}"
                            if val is not None:
                                try:
                                    txt += f"\nValue: ${float(val):,.0f}"
                                except Exception:
                                    txt += f"\nValue: {val}"
                            if cost is not None:
                                try:
                                    txt += f"\nPrice: ${float(cost):.2f}"
                                except Exception:
                                    txt += f"\nPrice: {cost}"
                            if pct is not None:
                                try:
                                    txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                except Exception:
                                    txt += f"\n% Holding Sold: {pct}"
                            sel.annotation.set_text(txt)
                        cur.connect("add", _on_add)
                        try:
                            self.status_var.set("Insider tooltips: hover enabled")
                        except Exception:
                            pass
                    except Exception:
                        try:
                            coll.set_picker(True)
                            disp_records = mapped_records
                            fig = ax.figure
                            annot = ax.annotate("", xy=(0, 0), xytext=(8, 8), textcoords="offset points")
                            annot.set_visible(False)
                            def _on_pick(ev):
                                inds = getattr(ev, "ind", [])
                                if not inds:
                                    return
                                i = int(inds[0])
                                if i < 0 or i >= len(disp_records):
                                    return
                                r = disp_records[i]
                                x = pts_x[i]
                                y = pts_y[i]
                                who = str(r.get("Who") or "").strip()
                                tx = str(r.get("Transaction") or "Sale").strip()
                                dt = r.get("Date")
                                sh = r.get("Shares")
                                val = r.get("Value")
                                cost = r.get("Cost")
                                pct = r.get("PercentSold")
                                txt = f"{who} ({tx})"
                                if dt is not None:
                                    txt += f"\nDate: {pd.to_datetime(dt)}"
                                if sh is not None:
                                    try:
                                        txt += f"\nShares: {int(float(sh)):,}"
                                    except Exception:
                                        txt += f"\nShares: {sh}"
                                if val is not None:
                                    try:
                                        txt += f"\nValue: ${float(val):,.0f}"
                                    except Exception:
                                        txt += f"\nValue: {val}"
                                if cost is not None:
                                    try:
                                        txt += f"\nPrice: ${float(cost):.2f}"
                                    except Exception:
                                        txt += f"\nPrice: {cost}"
                                if pct is not None:
                                    try:
                                        txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                    except Exception:
                                        txt += f"\n% Holding Sold: {pct}"
                                annot.set_text(txt)
                                annot.xy = (x, y)
                                annot.set_visible(True)
                                fig.canvas.draw_idle()
                            fig.canvas.mpl_connect("pick_event", _on_pick)
                            try:
                                self.status_var.set("Insider tooltips: click on bubble for details")
                            except Exception:
                                pass
                        except Exception:
                            pass
            if bool(self.show_marker_vlines_var.get()):
                print("Debug: plotting insider vlines", len(xs))
                for x in xs:
                    ax.axvline(x=x, color="#8c564b", alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="best")
        except Exception as e:
            print("Debug: plotting insider failed", str(e))
        # Only real sales plotted here; proposed handled separately
        try:
            self.status_var.set(f"Insider: fetched {total_count} total, mapped {mapped} in range, plotted {len(xs)} markers (scatter {len(pts_x)})")
            print("Insider debug:", {"symbol": symbol, "total": total_count, "in_range": mapped, "markers": len(xs), "scatter": len(pts_x)})
        except Exception:
            pass

    def _on_overlay_toggle(self):
        try:
            print("Debug: overlay toggled", {
                "earnings": bool(self.show_earnings_var.get()),
                "insider": bool(self.show_insider_var.get()),
                "insider_proposed": bool(self.show_insider_proposed_var.get()),
                "insider_tooltips": bool(self.insider_tooltips_var.get()),
                "volume_dot": bool(self.show_volume_dot_var.get()),
                "pe_eps": bool(self.show_pe_eps_var.get()),
                "full_moon": bool(self.show_full_moon_var.get()),
                "new_moon": bool(self.show_new_moon_var.get()),
                "first_quarter": bool(self.show_first_quarter_moon_var.get()),
                "last_quarter": bool(self.show_last_quarter_moon_var.get()),
            })
        except Exception:
            pass
        try:
            if self.last_fcst_df is not None:
                try:
                    print("Debug: overlay refresh path", {"mode": "forecast"})
                except Exception:
                    pass
                self.update_plot()
            else:
                try:
                    print("Debug: overlay refresh path", {"mode": "visualize"})
                except Exception:
                    pass
                self.visualize_data()
        except Exception as e:
            try:
                self.status_var.set(f"Overlay refresh failed: {e}")
            except Exception:
                pass
def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
    def _plot_earnings_markers(self, ax, df, time_col, target_col, symbol):
        if not symbol or df is None or df.empty:
            return
        try:
            import yfinance as yf
            import pandas as pd
        except Exception:
            return
        try:
            idx = pd.DatetimeIndex(pd.to_datetime(df[time_col]).dropna().sort_values())
            start = idx.min()
            end = idx.max()
        except Exception:
            return
        # Primary source: yfinance
        dates = pd.Series([], dtype='datetime64[ns]')
        surprise_map = {}
        try:
            tk = yf.Ticker(symbol.strip().upper())
            cal = tk.get_earnings_dates(limit=100)
            if cal is not None and len(cal) > 0:
                if hasattr(cal, 'columns') and 'Earnings Date' in list(cal.columns):
                    dates = pd.to_datetime(cal['Earnings Date'], utc=True).tz_convert(None)
                elif hasattr(cal, 'index'):
                    dates = pd.to_datetime(cal.index, utc=True).tz_convert(None)
            if not dates.empty:
                try:
                    print("Earnings (yfinance):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                except Exception:
                    pass
            if dates.empty:
                try:
                    cal2 = tk.calendar
                except Exception:
                    cal2 = None
                parsed_cal = []
                try:
                    if cal2:
                        if isinstance(cal2, dict):
                            ed = cal2.get('Earnings Date')
                            if ed:
                                if isinstance(ed, (list, tuple)):
                                    for d in ed:
                                        try:
                                            parsed_cal.append(pd.to_datetime(d, utc=True))
                                        except Exception:
                                            pass
                                else:
                                    try:
                                        parsed_cal.append(pd.to_datetime(ed, utc=True))
                                    except Exception:
                                        pass
                        else:
                            try:
                                dfc = pd.DataFrame(cal2).T
                                if 'Earnings Date' in dfc.columns:
                                    vals = dfc['Earnings Date'].values.tolist()
                                    for d in vals:
                                        try:
                                            parsed_cal.append(pd.to_datetime(d, utc=True))
                                        except Exception:
                                            pass
                            except Exception:
                                pass
                except Exception:
                    parsed_cal = []
                if parsed_cal:
                    try:
                        dates = pd.to_datetime(parsed_cal, utc=True).tz_convert(None)
                        print("Earnings (yfinance calendar):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                    except Exception:
                        pass
            try:
                edf = tk.earnings_dates
            except Exception:
                edf = None
            if edf is not None and hasattr(edf, 'index') and len(edf) > 0:
                try:
                    idx_dates = pd.to_datetime(edf.index, utc=True).tz_convert(None)
                    try:
                        print("Earnings (yfinance table):", [d.strftime("%Y-%m-%d") for d in idx_dates.tolist()])
                    except Exception:
                        pass
                    try:
                        if 'Surprise(%)' in edf.columns:
                            s = pd.to_numeric(edf['Surprise(%)'], errors='coerce')
                            for ts, val in zip(idx_dates.tolist(), s.tolist()):
                                if pd.notna(val):
                                    surprise_map[pd.to_datetime(ts)] = float(val)
                        if len(surprise_map) == 0:
                            est_col = None
                            rep_col = None
                            for c in ['EPS Estimate','Estimate','epsEstimate','estimatedEPS']:
                                if c in edf.columns:
                                    est_col = c
                                    break
                            for c in ['Reported EPS','Reported','reportedEPS','reported']:
                                if c in edf.columns:
                                    rep_col = c
                                    break
                            if est_col and rep_col:
                                est = pd.to_numeric(edf[est_col], errors='coerce')
                                rep = pd.to_numeric(edf[rep_col], errors='coerce')
                                surp = ((rep - est) / est * 100.0)
                                for ts, val in zip(idx_dates.tolist(), surp.tolist()):
                                    if pd.notna(val):
                                        surprise_map[pd.to_datetime(ts)] = float(val)
                    except Exception:
                        pass
                except Exception:
                    pass
        except Exception:
            pass
        # Fallback: MCP
        if dates.empty:
            try:
                import asyncio
                # Try local MCP server first (HTTP)
                try:
                    import json
                    import urllib.request
                    req = urllib.request.Request(
                        url="http://127.0.0.1:8088/tool/get_earnings_dates",
                        data=json.dumps({"symbol": symbol.strip().upper(), "limit": 100}).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST",
                    )
                    with urllib.request.urlopen(req, timeout=5) as resp:
                        body = resp.read()
                        obj = json.loads(body.decode("utf-8"))
                        data = obj.get("data", [])
                        parsed_local = []
                        parsed_surp = []
                        for item in data:
                            d = item.get("date") if isinstance(item, dict) else item
                            if d:
                                try:
                                    parsed_local.append(pd.to_datetime(d, utc=True))
                                except Exception:
                                    pass
                            if isinstance(item, dict):
                                for k in ["Surprise(%)", "surprise", "Surprise"]:
                                    if k in item and item[k] is not None:
                                        try:
                                            parsed_surp.append((pd.to_datetime(d, utc=True), float(item[k])))
                                            break
                                        except Exception:
                                            pass
                        if parsed_local:
                            dates = pd.to_datetime(parsed_local, utc=True).tz_convert(None)
                            try:
                                print("Earnings (MCP HTTP):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                            except Exception:
                                pass
                        if parsed_surp:
                            try:
                                for ts, val in parsed_surp:
                                    surprise_map[pd.to_datetime(ts).tz_convert(None)] = val
                            except Exception:
                                pass
                        if parsed_surp:
                            try:
                                for ts, val in parsed_surp:
                                    surprise_map[pd.to_datetime(ts).tz_convert(None)] = val
                                print("Debug: earnings surprises from MCP HTTP", {"count": len(parsed_surp)})
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    from spoonos_stock_agent import MCPClient
                except ImportError:
                    candidates = [
                        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spoon-core-jue")),
                        r"C:\\Users\\juesh\\jules\\spoon-core-jue",
                        os.path.expanduser(r"~\\jules\\spoon-core-jue"),
                    ]
                    for p in candidates:
                        if os.path.isdir(p) and p not in sys.path:
                            sys.path.append(p)
                    from spoonos_stock_agent import MCPClient
                async def fetch():
                    client = MCPClient(server_name="stock-mcp")
                    tools = [
                        "get_earnings_dates",
                        "get_stock_earnings_calendar",
                        "get_earnings_calendar",
                        "get_company_events",
                    ]
                    for t in tools:
                        try:
                            res = await client.call_tool(t, {"symbol": symbol.strip().upper()})
                            if isinstance(res, dict):
                                data = res.get("data")
                                if data:
                                    return data
                        except Exception:
                            continue
                    return []
                raw = asyncio.run(fetch())
                parsed = []
                if isinstance(raw, list):
                    for item in raw:
                        if isinstance(item, dict):
                            for k in ["date", "earningsDate", "reportedDate", "datetime", "time"]:
                                if k in item and item[k]:
                                    try:
                                        parsed.append(pd.to_datetime(item[k], utc=True))
                                        break
                                    except Exception:
                                        pass
                        else:
                            try:
                                parsed.append(pd.to_datetime(item, utc=True))
                            except Exception:
                                pass
                if parsed:
                    dates = pd.to_datetime(parsed, utc=True).tz_convert(None)
                    try:
                        print("Earnings (MCP client):", [d.strftime("%Y-%m-%d") for d in dates.tolist()])
                    except Exception:
                        pass
                try:
                    self.status_var.set(f"Earnings (MCP): {len(dates)} dates fetched")
                except Exception:
                    pass
            except Exception:
                pass
        total = len(dates)
        dates_full = dates.copy()
        dates = dates[(dates >= start) & (dates <= end)]
        if dates.empty:
            try:
                self.status_var.set(f"Earnings: {total} fetched, 0 in current range")
            except Exception:
                pass
            return
        def adjust(dt):
            i = idx.get_indexer([pd.to_datetime(dt)], method='pad')
            if len(i) and i[0] >= 0:
                return idx[i[0]]
            return None
        df_map = df.copy()
        df_map[time_col] = pd.to_datetime(df_map[time_col])
        df_map = df_map.dropna(subset=[time_col, target_col]) if target_col in df_map.columns else df_map.dropna(subset=[time_col])
        xs = []
        ys = []
        surprises = []
        for d in dates:
            a = adjust(d)
            if a is not None:
                try:
                    yv = df_map.loc[df_map[time_col] == a, target_col].iloc[0]
                except Exception:
                    yv = None
                xs.append(a)
                ys.append(yv)
                try:
                    dn = pd.to_datetime(d)
                    if getattr(dn, 'tzinfo', None) is not None:
                        dn = dn.tz_convert(None)
                    sv = surprise_map.get(pd.to_datetime(dn))
                except Exception:
                    sv = None
                surprises.append(sv)
        if xs:
            try:
                # Scatter only for points with y-values
                pts_x = [x for x, y in zip(xs, ys) if y is not None]
                pts_y = [y for y in ys if y is not None]
                if pts_x:
                    ax.scatter(pts_x, pts_y, s=30, color="#ff9900", label="Earnings")
                    try:
                        ann = [(x, y, s) for x, y, s in zip(xs, ys, surprises) if y is not None and s is not None]
                        for x, y, s in ann:
                            offs = 10 if s >= 0 else -12
                            ax.annotate(f"{s:.2f}%", (x, y), xytext=(0, offs), textcoords="offset points", color=("#2ca02c" if s >= 0 else "#d62728"))
                    except Exception:
                        pass
                if bool(self.show_marker_vlines_var.get()):
                    for x in xs:
                        ax.axvline(x=x, color="#ff0000", alpha=0.8)
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="best")
                try:
                    self.status_var.set(f"Earnings: plotted {len(xs)} markers (scatter {len(pts_x)})")
                except Exception:
                    pass
            except Exception:
                pass
        try:
            upcoming = dates_full[dates_full > end]
            if len(upcoming) > 0:
                next_dt = pd.to_datetime(upcoming.min())
                try:
                    ax.axvline(x=next_dt, color="#ff0000", alpha=0.8, linestyle="--")
                except Exception:
                    pass
                try:
                    from matplotlib.dates import date2num
                    l, r = ax.get_xlim()
                    nx = date2num(next_dt)
                    if nx > r:
                        ax.set_xlim(l, nx)
                except Exception:
                    pass
        except Exception:
            pass

    def _fetch_insider_sales(self, symbol):
        try:
            import yfinance as yf
            import pandas as pd
            import requests
            from io import StringIO
        except Exception:
            return None
        df = None
        try:
            ticker = yf.Ticker(symbol.strip().upper())
            df = ticker.insider_transactions
        except Exception:
            df = None
        if df is None or len(df) == 0:
            # Finviz fallback (Sale entries)
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                url = f"https://finviz.com/quote.ashx?t={symbol.strip().upper()}&p=d"
                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                import pandas as pd
                tables = pd.read_html(StringIO(resp.text))
                target = None
                for table in tables:
                    if isinstance(table.columns, pd.MultiIndex):
                        table.columns = table.columns.get_level_values(-1)
                    if 'Insider Trading' in table.columns:
                        target = table
                        break
                if target is None:
                    return None
                rename_map = {
                    'Insider Trading': 'Who',
                    'Relationship': 'Relationship',
                    'Date': 'Date',
                    'Transaction': 'TransactionText',
                    'Cost': 'Cost',
                    '#Shares': 'Shares',
                    '# Shares': 'Shares',
                    'Value ($)': 'Value'
                }
                df = target.rename(columns={k: v for k, v in rename_map.items() if k in target.columns})
                required_cols = {'Date', 'Who', 'TransactionText', 'Shares', 'Value', 'Cost'}
                if not required_cols.issubset(df.columns):
                    return None
                df['TransactionText'] = df['TransactionText'].astype(str)
                df = df[df['TransactionText'].str.contains("Sale", case=False, na=False)].copy()
                if df.empty:
                    return None
                df['Date'] = pd.to_datetime(df['Date'], format="%b %d '%y", errors='coerce')
                df['Shares'] = pd.to_numeric(df['Shares'].astype(str).str.replace(',', ''), errors='coerce')
                df['Value'] = pd.to_numeric(df['Value'].astype(str).str.replace(',', ''), errors='coerce')
                df['Cost'] = pd.to_numeric(df['Cost'].astype(str).str.replace(',', ''), errors='coerce')
                df.dropna(subset=['Date', 'Shares', 'Value', 'Cost'], inplace=True)
            except Exception:
                return None
            return df[['Date', 'Who', 'TransactionText', 'Cost', 'Shares', 'Value']].sort_values('Date')
        # yfinance format
        try:
            import pandas as pd
            df = df.reset_index()
            rename_map = {
                'Start Date': 'Date',
                'Insider': 'Who',
                'Transaction': 'TransactionType',
                'Text': 'TransactionText',
                'Value': 'TotalValue',
                '#Shares': 'Shares'
            }
            df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
            # Filter Sales
            cols = [c for c in ('TransactionType', 'TransactionText', 'Transaction') if c in df.columns]
            if not cols:
                return None
            sale_mask = pd.Series(False, index=df.index)
            for c in cols:
                sale_mask |= df[c].astype(str).str.contains("Sale", case=False, na=False)
            sdf = df[sale_mask].copy()
            if sdf.empty:
                return None
            sdf['Date'] = pd.to_datetime(sdf['Date'])
            sdf['Shares'] = pd.to_numeric(sdf['Shares'], errors='coerce')
            if 'TotalValue' in sdf.columns:
                sdf['TotalValue'] = pd.to_numeric(sdf['TotalValue'], errors='coerce')
            # Derive Cost and Value
            sdf = sdf[sdf['Shares'] > 0]
            if 'TotalValue' in sdf.columns:
                sdf['Cost'] = sdf['TotalValue'] / sdf['Shares']
                sdf.rename(columns={'TotalValue': 'Value'}, inplace=True)
            else:
                # If per-share cost present, compute Value
                if 'Cost' in sdf.columns:
                    sdf['Value'] = sdf['Cost'] * sdf['Shares']
            sdf = sdf.dropna(subset=['Date', 'Shares'])
            sdf = sdf.sort_values('Date')
            return sdf[['Date', 'Who', 'TransactionText' if 'TransactionText' in sdf.columns else 'Transaction', 'Cost', 'Shares', 'Value']].rename(columns={'TransactionText': 'Transaction'})
        except Exception:
            return None

    def _plot_insider_sales(self, ax, df, time_col, target_col, symbol):
        print("Debug: _plot_insider_sales called", {"symbol": symbol, "time_col": time_col, "target_col": target_col, "df_rows": 0 if df is None else len(df)})
        if df is None or df.empty:
            return
        if not symbol:
            try:
                if 'unique_id' in df.columns:
                    vals = [str(v).strip() for v in df['unique_id'].dropna().tolist() if str(v).strip()]
                    symbol = vals[0] if vals else symbol
            except Exception:
                pass
        symbol = (symbol or "").strip().upper()
        if not symbol:
            try:
                self.status_var.set("Insider: no symbol set; skip")
            except Exception:
                pass
            return
        try:
            print("Debug: fetching insider sales for", symbol)
            sales = self._fetch_insider_sales(symbol)
            try:
                print("Debug: fetched insider rows", 0 if sales is None else len(sales))
            except Exception:
                pass
        except Exception as e:
            try:
                self.status_var.set(f"Insider: error fetching for {symbol}: {e}")
            except Exception:
                pass
            sales = None
        if sales is None or sales.empty:
            try:
                self.status_var.set(f"Insider: no sales for {symbol}")
            except Exception:
                pass
            return
        total_count = len(sales)
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col])
        df_sorted = df_sorted.dropna(subset=[time_col])
        df_sorted = df_sorted.sort_values(time_col)
        idx = pd.DatetimeIndex(df_sorted[time_col].values)
        start = idx.min()
        end = idx.max()
        sales_full = sales.copy()
        sales = sales[(sales_full['Date'] >= start) & (sales_full['Date'] <= end)]
        if sales.empty:
            try:
                self.status_var.set(f"Insider: fetched {total_count} total, 0 in range")
            except Exception:
                pass
            return
        def adjust(dt):
            i = idx.get_indexer([pd.to_datetime(dt)], method='pad')
            if len(i) and i[0] >= 0:
                return i[0]
            return None
        xs = []
        ys = []
        sizes = []
        vals = pd.to_numeric(sales.get('Value', pd.Series([0]*len(sales))), errors='coerce').fillna(0.0)
        vmax = float(vals.max()) if len(vals) else 0.0
        mapped = 0
        mapped_records = []
        for _, row in sales.iterrows():
            pos = adjust(row['Date'])
            if pos is not None and pos >= 0 and pos < len(df_sorted):
                ts_adj = idx[pos]
                xs.append(ts_adj)
                try:
                    yv = (pd.to_numeric(df_sorted[target_col], errors='coerce') if target_col in df_sorted.columns else pd.to_numeric(df_sorted['close'], errors='coerce')).iloc[pos]
                except Exception:
                    yv = None
                ys.append(yv)
                if vmax > 0:
                    sizes.append(50 + 400 * (row.get('Value', 0.0) / vmax))
                else:
                    sizes.append(80)
                mapped += 1
                try:
                    rec = {
                        "Who": row.get("Who"),
                        "Transaction": row.get("TransactionText") if "TransactionText" in sales.columns else (row.get("Transaction") if "Transaction" in sales.columns else row.get("TransactionType")),
                        "Date": row.get("Date"),
                        "Shares": row.get("Shares"),
                        "Value": row.get("Value") if "Value" in sales.columns else row.get("TotalValue"),
                        "Cost": row.get("Cost"),
                    }
                    pct = None
                    for kc in ["SharesOwned","Owned","Holdings","Ownership","%Owned","Percent"]:
                        v = row.get(kc)
                        try:
                            if v is not None:
                                if kc in ["Ownership","%Owned","Percent"] and isinstance(v, str):
                                    vs = str(v).strip().strip("%")
                                    pct = float(vs)
                                    break
                                sv = float(row.get("Shares")) if row.get("Shares") is not None else None
                                ov = float(v)
                                if sv is not None and ov and ov != 0:
                                    pct = 100.0 * sv / ov
                                    break
                        except Exception:
                            continue
                    rec["PercentSold"] = pct
                    mapped_records.append(rec)
                except Exception:
                    pass
        pts_x = [x for x, y in zip(xs, ys) if y is not None]
        pts_y = [y for y in ys if y is not None]
        pts_s = [s for s, y in zip(sizes, ys) if y is not None]
        try:
            if pts_x:
                print("Debug: plotting insider scatter points", len(pts_x))
                coll = ax.scatter(pts_x, pts_y, s=pts_s, color="#8c564b", edgecolors="black", alpha=0.8, label="Insider Sales")
                if bool(self.insider_tooltips_var.get()):
                    try:
                        import mplcursors
                        disp_records = mapped_records
                        cur = mplcursors.cursor(coll, hover=True)
                        def _on_add(sel):
                            try:
                                i = int(sel.index)
                            except Exception:
                                return
                            if i < 0 or i >= len(disp_records):
                                return
                            r = disp_records[i]
                            who = str(r.get("Who") or "").strip()
                            tx = str(r.get("Transaction") or "Sale").strip()
                            dt = r.get("Date")
                            sh = r.get("Shares")
                            val = r.get("Value")
                            cost = r.get("Cost")
                            pct = r.get("PercentSold")
                            txt = f"{who} ({tx})"
                            if dt is not None:
                                txt += f"\nDate: {pd.to_datetime(dt)}"
                            if sh is not None:
                                try:
                                    txt += f"\nShares: {int(float(sh)):,}"
                                except Exception:
                                    txt += f"\nShares: {sh}"
                            if val is not None:
                                try:
                                    txt += f"\nValue: ${float(val):,.0f}"
                                except Exception:
                                    txt += f"\nValue: {val}"
                            if cost is not None:
                                try:
                                    txt += f"\nPrice: ${float(cost):.2f}"
                                except Exception:
                                    txt += f"\nPrice: {cost}"
                            if pct is not None:
                                try:
                                    txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                except Exception:
                                    txt += f"\n% Holding Sold: {pct}"
                            sel.annotation.set_text(txt)
                        cur.connect("add", _on_add)
                    except Exception:
                        try:
                            coll.set_picker(True)
                            disp_records = mapped_records
                            fig = ax.figure
                            annot = ax.annotate("", xy=(0, 0), xytext=(8, 8), textcoords="offset points")
                            annot.set_visible(False)
                            def _on_pick(ev):
                                inds = getattr(ev, "ind", [])
                                if not inds:
                                    return
                                i = int(inds[0])
                                if i < 0 or i >= len(disp_records):
                                    return
                                r = disp_records[i]
                                x = pts_x[i]
                                y = pts_y[i]
                                who = str(r.get("Who") or "").strip()
                                tx = str(r.get("Transaction") or "Sale").strip()
                                dt = r.get("Date")
                                sh = r.get("Shares")
                                val = r.get("Value")
                                cost = r.get("Cost")
                                pct = r.get("PercentSold")
                                txt = f"{who} ({tx})"
                                if dt is not None:
                                    txt += f"\nDate: {pd.to_datetime(dt)}"
                                if sh is not None:
                                    try:
                                        txt += f"\nShares: {int(float(sh)):,}"
                                    except Exception:
                                        txt += f"\nShares: {sh}"
                                if val is not None:
                                    try:
                                        txt += f"\nValue: ${float(val):,.0f}"
                                    except Exception:
                                        txt += f"\nValue: {val}"
                                if cost is not None:
                                    try:
                                        txt += f"\nPrice: ${float(cost):.2f}"
                                    except Exception:
                                        txt += f"\nPrice: {cost}"
                                if pct is not None:
                                    try:
                                        txt += f"\n% Holding Sold: {float(pct):.2f}%"
                                    except Exception:
                                        txt += f"\n% Holding Sold: {pct}"
                                annot.set_text(txt)
                                annot.xy = (x, y)
                                annot.set_visible(True)
                                fig.canvas.draw_idle()
                            fig.canvas.mpl_connect("pick_event", _on_pick)
                        except Exception:
                            pass
            if bool(self.show_marker_vlines_var.get()):
                print("Debug: plotting insider vlines", len(xs))
                for x in xs:
                    ax.axvline(x=x, color="#8c564b", alpha=0.25)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, loc="best")
        except Exception as e:
            print("Debug: plotting insider failed", str(e))
        try:
            upcoming = sales_full[sales_full['Date'] > end]
            if len(upcoming) > 0:
                for dt in upcoming['Date'].tolist():
                    try:
                        ax.axvline(x=pd.to_datetime(dt), color="#ff0000", alpha=0.8, linestyle="--")
                    except Exception:
                        continue
                try:
                    from matplotlib.dates import date2num
                    l, r = ax.get_xlim()
                    mx = date2num(pd.to_datetime(upcoming['Date'].max()))
                    if mx > r:
                        ax.set_xlim(l, mx)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.status_var.set(f"Insider: fetched {total_count} total, mapped {mapped} in range, plotted {len(xs)} markers (scatter {len(pts_x)})")
            print("Insider debug:", {"symbol": symbol, "total": total_count, "in_range": mapped, "markers": len(xs), "scatter": len(pts_x)})
        except Exception:
            pass
    def _plot_insider_proposed_sales(self, ax, df, time_col, target_col, symbol):
        try:
            print("Debug: _plot_insider_proposed_sales called", {"symbol": symbol, "time_col": time_col, "target_col": target_col, "df_rows": 0 if df is None else len(df)})
        except Exception:
            pass
        if ax is None or df is None or df.empty or not symbol:
            return
        try:
            sales = self._fetch_insider_sales((symbol or "").strip().upper())
        except Exception:
            sales = None
        if sales is None or sales.empty:
            return
        df_sorted = df.copy()
        df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors="coerce")
        df_sorted = df_sorted.dropna(subset=[time_col]).sort_values(time_col)
        idx = pd.DatetimeIndex(df_sorted[time_col].values)
        if len(idx) == 0:
            return
        start = idx.min()
        end = idx.max()
        try:
            txt_col = 'TransactionText' if 'TransactionText' in sales.columns else ('Transaction' if 'Transaction' in sales.columns else None)
        except Exception:
            txt_col = None
        if txt_col is None:
            return
        try:
            texts = sales[txt_col].astype(str)
            key_any = texts.str.contains("Proposed|Planned|Plan|Intend|Filed|Form 144|10b5-1", case=False, na=False)
            sale_any = texts.str.contains("Sale", case=False, na=False)
            proposed = sales[key_any & sale_any].copy()
        except Exception:
            proposed = None
        if proposed is None or proposed.empty:
            return
        try:
            proposed['Date'] = pd.to_datetime(proposed['Date'], errors="coerce")
            proposed = proposed.dropna(subset=['Date'])
        except Exception:
            return
        in_range = proposed[(proposed['Date'] >= start) & (proposed['Date'] <= end)]
        try:
            for dt in in_range['Date'].tolist():
                try:
                    ax.axvline(x=pd.to_datetime(dt), color="#ff0000", alpha=0.8, linestyle="--")
                except Exception:
                    continue
            try:
                print("Debug: proposed insider vlines", {"count": len(in_range)})
            except Exception:
                pass
        except Exception:
            pass
    def _plot_volume_dot_overlay(self, ax, df, time_col, target_col):
        try:
            print("Debug: _plot_volume_dot_overlay called", {
                "enabled": bool(self.show_volume_dot_var.get()) if hasattr(self, "show_volume_dot_var") else None,
                "time_col": time_col,
                "target_col": target_col,
                "df_rows": 0 if df is None else len(df),
                "cols": [] if df is None else list(df.columns),
            })
            try:
                import sys
                sys.stdout.flush()
            except Exception:
                pass
        except Exception:
            pass
        if ax is None or df is None or df.empty or not time_col:
            return
        dfx = df.copy()
        dfx[time_col] = pd.to_datetime(dfx[time_col], errors="coerce")
        dfx = dfx.dropna(subset=[time_col])
        # Resolve columns robustly across casing and common variants
        candidates_close = [target_col, 'close', 'Close', 'Adj Close']
        close_col = next((c for c in candidates_close if c and c in dfx.columns), None)
        candidates_open = ['open', 'Open']
        open_col = next((c for c in candidates_open if c in dfx.columns), None)
        candidates_vol = ['volume', 'Volume', 'vol', 'Vol']
        vol_col = next((c for c in candidates_vol if c in dfx.columns), None)
        try:
            print("Debug: volume dot columns resolved", {"close_col": close_col, "open_col": open_col, "vol_col": vol_col})
        except Exception:
            pass
        if close_col is None or open_col is None or vol_col is None:
            try:
                self.status_var.set("Volume dot: required columns missing (need open/close/volume)")
            except Exception:
                pass
            return
        dfx[close_col] = pd.to_numeric(dfx[close_col], errors="coerce")
        dfx[open_col] = pd.to_numeric(dfx[open_col], errors="coerce")
        dfx[vol_col] = pd.to_numeric(dfx[vol_col], errors="coerce")
        dfx = dfx.dropna(subset=[close_col, open_col, vol_col])
        try:
            print("Debug: volume dot row counts", {"rows_after_dropna": len(dfx)})
        except Exception:
            pass
        x = dfx[time_col]
        y = dfx[close_col]
        try:
            ax.plot(x, y, color="midnightblue", linewidth=1.3, alpha=0.85, label="Close trend")
        except Exception:
            pass
        try:
            colors = ['green' if float(c) > float(o) else 'red' for c, o in zip(dfx[close_col].tolist(), dfx[open_col].tolist())]
            try:
                gc = sum(1 for c in colors if c == 'green')
                rc = sum(1 for c in colors if c == 'red')
                print("Debug: volume dot colors", {"green": gc, "red": rc})
            except Exception:
                pass
        except Exception:
            colors = ['green'] * len(dfx)
        try:
            vols = []
            for vv in dfx[vol_col].tolist():
                try:
                    vols.append(float(vv))
                except Exception:
                    vols.append(float("nan"))
            # Filter NaNs
            vols_clean = [v for v in vols if v == v]
            vmin = min(vols_clean) if vols_clean else 0.0
            vmax = max(vols_clean) if vols_clean else 0.0
            try:
                print("Debug: volume dot sizes range", {"vmin": vmin, "vmax": vmax})
            except Exception:
                pass
            sizes = []
            lo, hi = 10.0, 80.0
            if vmin == vmax:
                sizes = [lo] * len(dfx)
            else:
                rng = (vmax - vmin)
                for v in vols:
                    if v != v:
                        sizes.append(lo)
                    else:
                        sizes.append(lo + (v - vmin) / rng * (hi - lo))
            try:
                print("Debug: volume dot sizes sample", {"count": int(len(sizes)), "min": float(min(sizes) if sizes else 0.0), "max": float(max(sizes) if sizes else 0.0)})
            except Exception:
                pass
        except Exception:
            sizes = [10.0] * len(dfx)
        try:
            sc = ax.scatter(x, y, s=sizes, c=colors, alpha=0.6, edgecolors="black", linewidth=0.5, label="Volume dots")
            try:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, loc="best")
            except Exception:
                pass
            try:
                self.status_var.set(f"Volume dot: plotted {len(dfx)} points")
                print("Debug: volume dot plotted", {"points": len(dfx)})
            except Exception:
                pass
            try:
                ax.figure.canvas.draw_idle()
            except Exception:
                pass
        except Exception:
            pass

    def _on_overlay_toggle(self):
        try:
            print("Debug: overlay toggled", {
                "earnings": bool(self.show_earnings_var.get()),
                "insider": bool(self.show_insider_var.get()),
                "insider_proposed": bool(self.show_insider_proposed_var.get()),
                "insider_tooltips": bool(self.insider_tooltips_var.get()),
                "full_moon": bool(self.show_full_moon_var.get()),
                "new_moon": bool(self.show_new_moon_var.get()),
                "first_quarter": bool(self.show_first_quarter_moon_var.get()),
                "last_quarter": bool(self.show_last_quarter_moon_var.get()),
            })
        except Exception:
            pass
        # Re-render based on current context
        try:
            if self.last_fcst_df is not None:
                self.update_plot()
            else:
                self.visualize_data()
        except Exception as e:
            try:
                self.status_var.set(f"Overlay refresh failed: {e}")
            except Exception:
                pass
