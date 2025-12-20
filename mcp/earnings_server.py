import json
from typing import List, Dict, Any

from flask import Flask, request, jsonify

app = Flask(__name__)


def fetch_earnings_dates(symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
    try:
        import yfinance as yf
        import pandas as pd
    except Exception:
        return []
    try:
        tk = yf.Ticker(symbol.strip().upper())
        cal = tk.get_earnings_dates(limit=limit)
        if cal is None or len(cal) == 0:
            return []
        # Normalize to list of dicts with 'date'
        dates = []
        if hasattr(cal, 'columns') and 'Earnings Date' in list(cal.columns):
            ser = pd.to_datetime(cal['Earnings Date'], utc=True).tz_convert(None)
            for d in ser.tolist():
                dates.append({"date": d.isoformat()})
        elif hasattr(cal, 'index'):
            ser = pd.to_datetime(cal.index, utc=True).tz_convert(None)
            for d in ser.tolist():
                dates.append({"date": d.isoformat()})
        return dates
    except Exception:
        return []


@app.route("/tool/get_earnings_dates", methods=["POST"])
def tool_get_earnings_dates():
    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        payload = {}
    symbol = (payload.get("symbol") or "").strip().upper()
    limit = int(payload.get("limit") or 100)
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    data = fetch_earnings_dates(symbol, limit=limit)
    return jsonify({"data": data})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8088)
