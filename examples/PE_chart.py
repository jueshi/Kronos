import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

import os, sys, importlib.util
def _import_sec_api_cache():
    # Prefer local module
    try:
        here = os.path.join(os.path.dirname(__file__), "sec_api_cache.py")
        if os.path.isfile(here):
            spec = importlib.util.spec_from_file_location("sec_api_cache", here)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        pass
    home = os.path.expanduser("~")
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
    import sec_api_cache
    return sec_api_cache
sec_api_cache = _import_sec_api_cache()
def _prefer_external_sec_cache():
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
                print("Debug: using external SEC cache dir", {"dir": sec_dir})
                break
    except Exception:
        pass
_prefer_external_sec_cache()
def _ensure_sec_env():
    try:
        # Prefer project .env if present
        root_env = os.path.join(os.path.dirname(__file__), "..", ".env")
        local_env = os.path.join(os.path.dirname(__file__), ".env")
        paths = [root_env, local_env]
        for p in paths:
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        for line in f:
                            s = line.strip()
                            if s.startswith("SEC_EDGAR_EMAIL="):
                                v = s.split("=", 1)[1].strip().strip('"').strip("'")
                                if v:
                                    os.environ.setdefault("SEC_EDGAR_EMAIL", v)
                            if s.startswith("SEC_API_KEY="):
                                v = s.split("=", 1)[1].strip().strip('"').strip("'")
                                if v:
                                    os.environ.setdefault("SEC_API_KEY", v)
                except Exception:
                    pass
    except Exception:
        pass
_ensure_sec_env()
try:
    print("Debug: using sec_api_cache module", {"path": getattr(sec_api_cache, "__file__", None)})
except Exception:
    pass

def _seed_cik_cache(ticker):
    try:
        import shutil
        home = os.path.expanduser("~")
        bases = [
            os.path.join(home, "OneDrive", "Documents", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "github", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "windsurf", "stock_charts_10k10q"),
            os.path.join(home, "OneDrive", "Documents", "stock_charts_10k-10q0", "stock_charts_10k10q"),
        ]
        src = None
        for base in bases:
            candidate = os.path.join(base, "sec_cache", "cik_lookups", f"{ticker.upper()}.txt")
            if os.path.isfile(candidate):
                src = candidate
                break
        if src:
            from pathlib import Path
            dst_dir = Path(getattr(sec_api_cache, "CACHE_DIR", Path("sec_cache"))) / "cik_lookups"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = str(dst_dir / f"{ticker.upper()}.txt")
            shutil.copyfile(src, dst)
            print("Debug: seeded CIK cache from OneDrive", {"src": src, "dst": dst})
    except Exception as e:
        print("Debug: seed CIK cache failed", {"error": str(e)})

def _seed_cik_from_company_tickers_cache(ticker):
    try:
        import hashlib
        from pathlib import Path
        base_dir = Path(getattr(sec_api_cache, "CACHE_DIR", Path("sec_cache")))
        url = "https://www.sec.gov/files/company_tickers.json"
        h = hashlib.md5(url.encode()).hexdigest()
        p = base_dir / "company_tickers" / f"{h}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            data = payload.get("data")
            if isinstance(data, dict):
                for _, company in data.items():
                    if str(company.get("ticker", "")).upper() == ticker.upper():
                        cik = str(company.get("cik_str")).zfill(10)
                        dst_dir = base_dir / "cik_lookups"
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        dst = dst_dir / f"{ticker.upper()}.txt"
                        with open(dst, "w") as wf:
                            wf.write(cik)
                        print("Debug: seeded CIK from cached company_tickers", {"dst": str(dst), "cik": cik})
                        return cik
            elif isinstance(data, list):
                for rec in data:
                    tk = str(rec.get("ticker", "")).upper()
                    if tk == ticker.upper():
                        cik = str(rec.get("cik_str")).zfill(10)
                        dst_dir = base_dir / "cik_lookups"
                        dst_dir.mkdir(parents=True, exist_ok=True)
                        dst = dst_dir / f"{ticker.upper()}.txt"
                        with open(dst, "w") as wf:
                            wf.write(cik)
                        print("Debug: seeded CIK from cached company_tickers", {"dst": str(dst), "cik": cik})
                        return cik
    except Exception:
        pass
    return None
def _inject_company_ticker_cache(ticker):
    try:
        from pathlib import Path
        base_dir = Path(getattr(sec_api_cache, "CACHE_DIR"))
        f = base_dir / "cik_lookups" / f"{ticker.upper()}.txt"
        if f.exists():
            with open(f, "r") as rf:
                cik = rf.read().strip()
            if cik:
                sec_api_cache.company_tickers_cache = {
                    "0": {"ticker": ticker.upper(), "cik_str": int(cik), "title": ticker.upper()}
                }
                from datetime import datetime as _dt
                sec_api_cache.company_tickers_last_update = _dt.now()
                print("Debug: injected in-memory company tickers cache", {"ticker": ticker.upper(), "cik": cik})
    except Exception:
        pass
def _seed_cik_fallback(ticker):
    try:
        from pathlib import Path
        mapping = {
            "AAPL": "0000320193",
            "AMD": "0000002488",
            "COST": "0000909832",
        }
        tk = ticker.upper()
        if tk in mapping:
            base_dir = Path(getattr(sec_api_cache, "CACHE_DIR"))
            dst_dir = base_dir / "cik_lookups"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f"{tk}.txt"
            if not dst.exists():
                with open(dst, "w") as wf:
                    wf.write(mapping[tk])
                print("Debug: seeded CIK from fallback mapping", {"ticker": tk, "cik": mapping[tk]})
    except Exception:
        pass
def _to_naive_datetime_index(index):
    """Ensure index is timezone-naive DatetimeIndex for safe comparisons."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _extract_eps_row(financials):
    if financials is None or financials.empty:
        return None
    candidates = ('Diluted EPS', 'Diluted EPS (USD)', 'Basic EPS', 'Basic EPS (USD)')
    for label in candidates:
        if label in financials.index:
            row = financials.loc[label].dropna()
            if not row.empty:
                numeric_row = pd.to_numeric(row, errors='coerce').dropna()
                if not numeric_row.empty:
                    return numeric_row
    return None


def _is_quarter_entry(form, frame, fp):
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


def _fetch_sec_eps_series(ticker):
    _seed_cik_cache(ticker)
    try:
        from pathlib import Path
        cache_dir = Path(getattr(sec_api_cache, "CACHE_DIR"))
        if not (cache_dir / "cik_lookups" / f"{ticker.upper()}.txt").exists():
            _seed_cik_from_company_tickers_cache(ticker)
        if not (cache_dir / "cik_lookups" / f"{ticker.upper()}.txt").exists():
            _seed_cik_fallback(ticker)
    except Exception:
        _seed_cik_from_company_tickers_cache(ticker)
        _seed_cik_fallback(ticker)
    try:
        _inject_company_ticker_cache(ticker)
        cik = sec_api_cache.get_company_cik(ticker.upper())
    except Exception as exc:
        print(f"无法获取 {ticker} 的CIK: {exc}")
        return None
    if not cik:
        return None
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    data = sec_api_cache.make_sec_request(url, "company_facts")
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
            if _is_quarter_entry(form, frame, fp):
                quarterly_entries.append((dt, val))
            else:
                annual_entries.append((dt, val))
    segments = []
    if quarterly_entries:
        quarterly_entries.sort(key=lambda x: x[0])
        q_idx = pd.DatetimeIndex([dt for dt, _ in quarterly_entries])
        q_vals = [val for _, val in quarterly_entries]
        q_series = pd.Series(q_vals, index=_to_naive_datetime_index(q_idx))
        q_series = q_series[~q_series.index.duplicated(keep='last')].sort_index()
        q_ttm = q_series.rolling(window=4, min_periods=4).sum().dropna()
        if not q_ttm.empty:
            segments.append(q_ttm)
    if annual_entries:
        annual_entries.sort(key=lambda x: x[0])
        a_idx = pd.DatetimeIndex([dt for dt, _ in annual_entries])
        a_vals = [val for _, val in annual_entries]
        a_series = pd.Series(a_vals, index=_to_naive_datetime_index(a_idx))
        a_series = a_series[~a_series.index.duplicated(keep='last')].sort_index()
        if not a_series.empty:
            segments.append(a_series)
    if not segments:
        return None
    combined = pd.concat(segments)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    try:
        print("Debug: SEC EPS series built", {"points": int(len(combined)), "start": str(combined.index.min()), "end": str(combined.index.max())})
    except Exception:
        pass
    return combined


def _build_eps_series(stock, ticker):
    eps_segments = []
    sec_eps = _fetch_sec_eps_series(ticker)
    if sec_eps is not None and not sec_eps.empty:
        eps_segments.append(sec_eps)
        try:
            print("Debug: using SEC EPS", {"points": int(len(sec_eps)), "start": str(sec_eps.index.min())})
        except Exception:
            pass
    financial_sources = (
        ('quarterly_financials', 'quarterly'),
        ('financials', 'annual'),
    )
    for attr, freq in financial_sources:
        financials = getattr(stock, attr, None)
        eps_row = _extract_eps_row(financials)
        if eps_row is None:
            continue
        idx = pd.to_datetime(eps_row.index)
        idx = _to_naive_datetime_index(idx)
        segment = pd.Series(eps_row.values, index=idx).sort_index()
        if freq == 'quarterly':
            segment = segment.rolling(window=4, min_periods=4).sum()
            segment = segment.dropna()
        if not segment.empty:
            eps_segments.append(segment)
    if not eps_segments:
        return None
    combined = pd.concat(eps_segments)
    combined = combined[~combined.index.duplicated(keep='first')]
    combined = combined.sort_index()
    return combined


def plot_pe_ratio_over_time(ticker, period='5y'):
    """
    获取股票历史数据，计算PE比率并绘制图表
    ticker: 股票代码，如 'AAPL', 'ALAB'
    period: 时间周期，如 '1y', '5y', 'max'
    """
    # 下载历史股价数据
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    hist.index = _to_naive_datetime_index(hist.index)
    
    eps_series = _build_eps_series(stock, ticker)
    if eps_series is None or eps_series.empty:
        print("未找到可靠的EPS历史，无法绘制PE图表。")
        return
    
    # 将EPS历史向前填充到每日数据（实际PE是TTM滚动计算，此处简化）
    hist['EPS'] = np.nan
    for date, eps in eps_series.items():
        hist.loc[hist.index >= date, 'EPS'] = eps
    
    # 填充EPS，计算PE = Close / EPS
    hist['EPS'] = hist['EPS'].ffill()
    hist['PE'] = hist['Close'] / hist['EPS']
    
    # 清理无效数据
    hist = hist.dropna(subset=['PE'])
    
    plot_pe(hist, ticker)

def plot_pe(df, ticker):
    fig, (ax_pe, ax_eps) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.15},
        constrained_layout=True,
    )
    pe_line, = ax_pe.plot(df.index, df['PE'], linewidth=2, label='Trailing PE Ratio', color='#1f77b4')
    ax_pe.set_ylabel('P/E Ratio', color=pe_line.get_color())
    ax_pe.tick_params(axis='y', labelcolor=pe_line.get_color())
    ax_pe.grid(True, alpha=0.3)

    ax_price = ax_pe.twinx()
    price_line, = ax_price.plot(df.index, df['Close'], linewidth=1.5, label='Close Price', color='#ff7f0e')
    ax_price.set_ylabel('Close Price', color=price_line.get_color())
    ax_price.tick_params(axis='y', labelcolor=price_line.get_color())

    lines = [pe_line, price_line]
    ax_pe.legend(lines, [line.get_label() for line in lines], loc='upper left')
    ax_pe.set_title(f'{ticker} PE & Price Over Time')

    ax_eps.plot(df.index, df['EPS'], color='#2ca02c', linewidth=1.8, label='Annualized EPS (TTM)')
    ax_eps.set_ylabel('EPS (TTM)')
    ax_eps.set_xlabel('Date')
    ax_eps.grid(True, alpha=0.3)
    latest_eps = df['EPS'].iloc[-1] if not df['EPS'].empty else np.nan
    if pd.notna(latest_eps):
        ax_eps.annotate(
            f'Latest: {latest_eps:.2f}',
            xy=(df.index[-1], latest_eps),
            xytext=(-10, 10),
            textcoords='offset points',
            ha='right',
            fontsize=9,
            color='#2ca02c',
        )
    ax_eps.legend(loc='upper left')

    plt.show()
    
    # 输出统计信息
    print(f"\n{ticker} PE Ratio 统计:")
    print(df['PE'].describe())

# 使用示例
if __name__ == "__main__":
    ticker = "aapl"  # 替换为你的目标股票，如 "ALAB" 或 "AMD"
    plot_pe_ratio_over_time(ticker, period='max')
