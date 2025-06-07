"""Nightly threshold auto-calibration

Usage (cron):
    # 02:30 server local time every day
    30 2 * * * cd /root/analytics-tool-v2 && venv/bin/python threshold_calibrator.py >> calibrator.log 2>&1

Algorithm
---------
1. Load realised trade log and historical scores for the last 30 days.
   The file paths are configurable via ENV or fallback defaults.
2. For each category (majors, memes) grid-search candidate long / short
   thresholds that maximise Sharpe ratio (majors) or hit-rate (memes).
3. Only accept new thresholds if performance >= 0.9 × previous Sharpe / hit-rate.
4. Persist updated values to `thresholds.yml` (same schema as analyzer).
5. Send a Telegram alert summarising the change.

The script is intentionally lightweight (<2s run-time on VPS).
"""
from __future__ import annotations

import os
import sys
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np

# --- Configuration ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
THRESHOLD_FILE = BASE_DIR / "thresholds.yml"
TRADE_LOG_PATH = Path(os.getenv("TRADE_LOG_PATH", BASE_DIR / "trade_log.csv"))
SCORE_LOG_PATH = Path(os.getenv("SCORE_LOG_PATH", BASE_DIR / "score_history.csv"))
LOOKBACK_DAYS = int(os.getenv("CAL_LOOKBACK_DAYS", 30))

# Fallback default thresholds if YAML missing
DEFAULT_THRESHOLDS = {
    "majors": {"long": 60, "short": 60},
    "memes": {"long": 65, "short": 55},
}

# Search grid for thresholds
GRID_VALUES = list(range(50, 81, 5))  # 50,55,60,65,70,75,80

# Telegram (optional)
try:
    from telegram_alerts import TelegramAlerts  # pragma: no cover
    TELEGRAM = TelegramAlerts()
except Exception:  # pragma: no cover
    TELEGRAM = None

# ---------------------------------------------------------------------------

def load_thresholds() -> Dict[str, Dict[str, int]]:
    if THRESHOLD_FILE.exists():
        with open(THRESHOLD_FILE, "r") as f:
            data = yaml.safe_load(f) or {}
        for cat, vals in DEFAULT_THRESHOLDS.items():
            data.setdefault(cat, vals)
        return data
    return DEFAULT_THRESHOLDS.copy()


def save_thresholds(new_cfg: Dict[str, Dict[str, int]]):
    THRESHOLD_FILE.write_text(yaml.safe_dump(new_cfg, sort_keys=False))


def sharpe(series: pd.Series) -> float:
    if series.std(ddof=0) == 0:
        return 0.0
    return series.mean() / series.std(ddof=0)


def evaluate_thresholds(df: pd.DataFrame, category: str, long_t: int, short_t: int) -> float:
    """Return performance metric (Sharpe majors, hit-rate memes)."""
    cat_df = df[df["category"] == category]
    if cat_df.empty:
        return 0.0

    if category == "majors":
        # trades executed when score >= threshold, pnl column already realised return
        mask = (cat_df["score"] >= cat_df["threshold"])
        metric = sharpe(cat_df.loc[mask, "pnl"])
    else:  # memes
        mask_short = cat_df["direction"].eq("SHORT") & (cat_df["score"] >= short_t)
        mask_long = cat_df["direction"].eq("LONG") & (cat_df["score"] >= long_t)
        mask = mask_short | mask_long
        if mask.sum() == 0:
            return 0.0
        metric = (cat_df.loc[mask, "pnl"] > 0).mean()  # hit-rate
    return float(metric)


def brute_search(df: pd.DataFrame, category: str) -> Tuple[int, int, float]:
    best_long, best_short, best_metric = 0, 0, -np.inf
    for long_t in GRID_VALUES:
        for short_t in GRID_VALUES:
            metric = evaluate_thresholds(df, category, long_t, short_t)
            if metric > best_metric:
                best_long, best_short, best_metric = long_t, short_t, metric
    return best_long, best_short, best_metric


def main() -> None:
    # Load data
    if not TRADE_LOG_PATH.exists():
        print(f"Trade log missing: {TRADE_LOG_PATH}")
        sys.exit(0)  # nothing to do
    trades = pd.read_csv(TRADE_LOG_PATH, parse_dates=["timestamp"])  # expected cols: timestamp, symbol, pnl, score, direction, category
    cutoff = datetime.utcnow() - timedelta(days=LOOKBACK_DAYS)
    trades = trades[trades["timestamp"] >= cutoff]
    if trades.empty:
        print("No trades in lookback window – aborting calibration")
        return

    current_cfg = load_thresholds()
    new_cfg = current_cfg.copy()
    summary: List[str] = []

    for cat in ["majors", "memes"]:
        best_long, best_short, best_metric = brute_search(trades, cat)
        # Evaluate current metric
        cur_long = current_cfg[cat]["long"]
        cur_short = current_cfg[cat]["short"]
        cur_metric = evaluate_thresholds(trades.assign(threshold=cur_long), cat, cur_long, cur_short)

        if best_metric >= 0.9 * cur_metric and (best_long != cur_long or best_short != cur_short):
            new_cfg[cat]["long"] = best_long
            new_cfg[cat]["short"] = best_short
            summary.append(f"{cat}: {cur_long}/{cur_short} → {best_long}/{best_short} (metric {cur_metric:.3f}→{best_metric:.3f})")

    if summary:
        save_thresholds(new_cfg)
        msg = "\n".join(["⚙️ Thresholds auto-calibrated:"] + summary)
        print(msg)
        if TELEGRAM:
            TELEGRAM.send_alert(msg)
    else:
        print("No threshold change – existing values retained.")


if __name__ == "__main__":
    main()
