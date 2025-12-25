#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import random
import requests
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# Simple .env loader
def _load_env():
    try:
        roots = [
            Path(__file__).resolve().parent.parent / ".env",
            Path(__file__).resolve().parent / ".env",
            Path.cwd() / ".env",
        ]
        for p in roots:
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        s = line.strip()
                        if not s or s.startswith("#") or "=" not in s:
                            continue
                        k, v = s.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if v and k not in os.environ:
                            os.environ[k] = v
    except Exception:
        pass

_load_env()

# Configurable cache dir via env
CACHE_DIR = Path(os.getenv("SEC_CACHE_DIR", "sec_cache"))
CACHE_DIR.mkdir(exist_ok=True)

CACHE_EXPIRATION = {
    "company_tickers": 7,
    "company_submissions": 1,
    "filing_document": 30,
}

MIN_DELAY = 10
MAX_RETRIES = 5
BASE_BACKOFF = 10
MAX_BACKOFF = 300

last_request_time = 0
company_tickers_cache = None
company_tickers_last_update = None

def get_headers():
    email = os.getenv("SEC_EDGAR_EMAIL", "") or "unknown@example.com"
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0 Safari/537.36",
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "From": email,
    }

def get_cache_path(url, cache_type):
    url_hash = hashlib.md5(url.encode()).hexdigest()
    sub = CACHE_DIR / cache_type
    sub.mkdir(exist_ok=True)
    return sub / f"{url_hash}.json"

def is_cache_valid(cache_path, cache_type):
    if not cache_path.exists():
        return False
    days = CACHE_EXPIRATION.get(cache_type, 1)
    mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=days)

def save_to_cache(url, data, cache_type):
    p = get_cache_path(url, cache_type)
    payload = {"url": url, "timestamp": datetime.now().isoformat(), "data": data}
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f)

def load_from_cache(url, cache_type):
    p = get_cache_path(url, cache_type)
    if not is_cache_valid(p, cache_type):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload["data"]
    except Exception:
        return None

def make_sec_request(url, cache_type, max_retries=MAX_RETRIES, force_refresh=False):
    global last_request_time
    if not force_refresh:
        cached = load_from_cache(url, cache_type)
        if cached is not None:
            return cached
    headers = get_headers()
    now = time.time()
    since = now - last_request_time
    if since < MIN_DELAY:
        wait = MIN_DELAY - since + random.uniform(1, 5)
        time.sleep(wait)
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                max_j = min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF)
                delay = max_j * 0.5 + max_j * 0.5 * random.random()
                time.sleep(delay)
            last_request_time = time.time()
            resp = requests.get(url, headers=headers, timeout=120)
            if resp.status_code == 200:
                if resp.headers.get("Content-Type", "").startswith("application/json"):
                    data = resp.json()
                else:
                    data = resp.text
                save_to_cache(url, data, cache_type)
                return data
            elif resp.status_code in (403, 429):
                time.sleep(MIN_DELAY * (2 if resp.status_code == 403 else 4))
            else:
                time.sleep(MIN_DELAY * 0.5)
        except Exception:
            pass
    return None

def get_company_tickers(force_refresh=False):
    global company_tickers_cache, company_tickers_last_update
    now = datetime.now()
    valid = (
        company_tickers_cache is not None
        and company_tickers_last_update is not None
        and (now - company_tickers_last_update).days < CACHE_EXPIRATION["company_tickers"]
        and not force_refresh
    )
    if valid:
        return company_tickers_cache
    url = "https://www.sec.gov/files/company_tickers.json"
    data = make_sec_request(url, "company_tickers", force_refresh=force_refresh)
    if data is not None:
        company_tickers_cache = data
        company_tickers_last_update = now
    return data

def _read_cached_company_tickers():
    try:
        url = "https://www.sec.gov/files/company_tickers.json"
        h = hashlib.md5(url.encode()).hexdigest()
        fp = CACHE_DIR / "company_tickers" / f"{h}.json"
        if fp.exists():
            with open(fp, "r", encoding="utf-8") as f:
                payload = json.load(f)
            return payload.get("data")
    except Exception:
        return None
    return None

def get_company_cik(ticker):
    cik_dir = CACHE_DIR / "cik_lookups"
    cik_dir.mkdir(exist_ok=True)
    p = cik_dir / f"{ticker.upper()}.txt"
    if p.exists():
        try:
            with open(p, "r") as f:
                v = f.read().strip()
                if v:
                    return v
        except Exception:
            pass
    # Env override e.g. SEC_CIK_AAPL=0000320193
    env_key = f"SEC_CIK_{ticker.upper()}"
    if os.getenv(env_key):
        val = os.getenv(env_key).strip()
        if val:
            try:
                with open(p, "w") as f:
                    f.write(val)
            except Exception:
                pass
            return val
    # Fallback mapping for common tickers
    fallback = {
        "AAPL": "0000320193",
        "AMD": "0000002488",
        "COST": "0000909832",
    }
    if ticker.upper() in fallback:
        val = fallback[ticker.upper()]
        try:
            with open(p, "w") as f:
                f.write(val)
        except Exception:
            pass
        return val
    # Try reading previously cached company_tickers file
    cached = _read_cached_company_tickers()
    if isinstance(cached, dict):
        for _, company in cached.items():
            if str(company.get("ticker", "")).upper() == ticker.upper():
                cik = str(company.get("cik_str")).zfill(10)
                try:
                    with open(p, "w") as f:
                        f.write(cik)
                except Exception:
                    pass
                return cik
    elif isinstance(cached, list):
        for rec in cached:
            tk = str(rec.get("ticker", "")).upper()
            if tk == ticker.upper():
                cik = str(rec.get("cik_str")).zfill(10)
                try:
                    with open(p, "w") as f:
                        f.write(cik)
                except Exception:
                    pass
                return cik
    companies = get_company_tickers()
    if not companies:
        return None
    if isinstance(companies, dict):
        for _, company in companies.items():
            if str(company.get("ticker", "")).upper() == ticker.upper():
                cik = str(company.get("cik_str")).zfill(10)
                try:
                    with open(p, "w") as f:
                        f.write(cik)
                except Exception:
                    pass
                return cik
    elif isinstance(companies, list):
        for rec in companies:
            tk = str(rec.get("ticker", "")).upper()
            if tk == ticker.upper():
                cik = str(rec.get("cik_str")).zfill(10)
                try:
                    with open(p, "w") as f:
                        f.write(cik)
                except Exception:
                    pass
                return cik
    return None

def get_company_submissions(cik):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    return make_sec_request(url, "company_submissions")
