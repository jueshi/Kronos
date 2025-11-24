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

        g1b = ttk.Frame(frame)
        g1b.pack(fill=tk.X, padx=8)
        self.impute_target_cb = ttk.Checkbutton(g1b, text="Impute Missing Target", variable=self.impute_target_var)
        self.impute_target_cb.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(g1b, text="Method").grid(row=0, column=1, sticky=tk.W)
        self.impute_method_cb = ttk.Combobox(g1b, textvariable=self.impute_method_var, values=["ffill","bfill","ffill_bfill","interpolate","interpolate_ffill_bfill"], width=12)
        self.impute_method_cb.grid(row=0, column=2, sticky=tk.W)
        self.auto_fix_ts_cb = ttk.Checkbutton(g1, text="Auto-fix timestamps (dedupe & fill gaps)", variable=self.auto_fix_ts_var)
        self.auto_fix_ts_cb.grid(row=0, column=6, sticky=tk.W)

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
        self.add_tooltip(self.h_entry, "Forecast horizon h (integer).")
        self.add_tooltip(self.display_start_entry, "Plot start time. Leave empty to auto.")
        self.add_tooltip(self.display_end_entry, "Plot end time. Leave empty to auto.")

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
            x = act_df[time_col]
            y = act_df[target_col]
            self._setup_hover(ax, [(x, y, "Actual")])
        except Exception:
            pass

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
        # Plot actual close
        if target_col in act_ext.columns:
            ax.plot(act_ext[time_col], act_ext[target_col], label="Actual", color="#1f77b4")
        elif 'close' in act_ext.columns:
            ax.plot(act_ext[time_col], act_ext['close'], label="Actual", color="#1f77b4")
        # Plot Kronos forecast close
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

def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
