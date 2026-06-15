"""
Quick EDA: load the training parquet and plot PM2.5 over time.

Produces ``reports/pm25_timeseries.png`` so you can eyeball seasonality
and long-term trends before modelling.

Run::

    python scripts/eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


def main() -> None:
    """Load parquet and save a PM2.5 time-series plot."""
    parquet_path = config.DATA_DIR / "training_data.parquet"
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run `python -m src.build_dataset` first.")
        sys.exit(1)

    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(
        df.index,
        df["pm2_5_mean"],
        linewidth=0.6,
        color="#4a90d9",
        alpha=0.7,
        label="Daily mean PM2.5",
    )

    # 30-day rolling mean overlay
    rolling30 = df["pm2_5_mean"].rolling(window=30, min_periods=15).mean()
    ax.plot(
        df.index,
        rolling30,
        linewidth=2,
        color="#d94a4a",
        label="30-day rolling mean",
    )

    ax.set_title(
        f"Daily PM2.5 — {config.CITY_NAME}  ({config.START_DATE} → {config.END_DATE})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("PM2.5 (μg/m³)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()

    output_path = config.REPORTS_DIR / "pm25_timeseries.png"
    fig.savefig(output_path, dpi=150)
    print(f"Saved time-series plot -> {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
