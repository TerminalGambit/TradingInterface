"""
Fetch historical BTC data from Binance.
Downloads OHLC candles instantly - no waiting for real-time collection.
"""

import requests
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"


def fetch_historical_candles(symbol: str = "BTCUSDT",
                              interval: str = "1m",
                              days: int = 7,
                              limit_per_request: int = 1000) -> dict:
    """
    Fetch historical OHLC candles from Binance.

    Args:
        symbol: Trading pair (default: BTCUSDT)
        interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
        days: Number of days of history to fetch
        limit_per_request: Max candles per API call (Binance max: 1000)

    Returns:
        Dict with 'opens', 'highs', 'lows', 'closes', 'volumes', 'timestamps'
    """
    all_candles = []

    # Calculate time range
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    print(f"Fetching {days} days of {interval} candles for {symbol}...")

    current_start = start_time
    request_count = 0

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "endTime": end_time,
            "limit": limit_per_request
        }

        try:
            response = requests.get(BINANCE_KLINES_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if not data:
                break

            all_candles.extend(data)
            request_count += 1

            # Move start time to after last candle
            last_close_time = data[-1][6]  # Close time of last candle
            current_start = last_close_time + 1

            print(f"\r  Fetched {len(all_candles)} candles...", end="", flush=True)

            if len(data) < limit_per_request:
                break  # No more data available

        except requests.RequestException as e:
            print(f"\nError fetching data: {e}")
            break

    print(f"\n  Total: {len(all_candles)} candles in {request_count} requests")

    if not all_candles:
        return {
            "opens": np.array([]),
            "highs": np.array([]),
            "lows": np.array([]),
            "closes": np.array([]),
            "volumes": np.array([]),
            "timestamps": []
        }

    # Parse candle data
    # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
    return {
        "opens": np.array([float(c[1]) for c in all_candles]),
        "highs": np.array([float(c[2]) for c in all_candles]),
        "lows": np.array([float(c[3]) for c in all_candles]),
        "closes": np.array([float(c[4]) for c in all_candles]),
        "volumes": np.array([float(c[5]) for c in all_candles]),
        "timestamps": [datetime.fromtimestamp(c[0]/1000).isoformat() for c in all_candles]
    }


def save_historical_data(candles: dict, filename: str = "historical_btc.json"):
    """Save fetched candles to JSON file."""
    data_dir = Path(__file__).parent / "training_data"
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / filename

    # Convert numpy arrays to lists for JSON
    save_data = {
        "opens": candles["opens"].tolist(),
        "highs": candles["highs"].tolist(),
        "lows": candles["lows"].tolist(),
        "closes": candles["closes"].tolist(),
        "volumes": candles["volumes"].tolist(),
        "timestamps": candles["timestamps"]
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f)

    print(f"Saved to {filepath}")
    return filepath


def load_historical_data(filename: str = "historical_btc.json") -> dict:
    """Load historical candles from JSON file."""
    data_dir = Path(__file__).parent / "training_data"
    filepath = data_dir / filename

    if not filepath.exists():
        return None

    with open(filepath, "r") as f:
        data = json.load(f)

    return {
        "opens": np.array(data["opens"]),
        "highs": np.array(data["highs"]),
        "lows": np.array(data["lows"]),
        "closes": np.array(data["closes"]),
        "volumes": np.array(data["volumes"]),
        "timestamps": data["timestamps"]
    }


if __name__ == "__main__":
    import sys
    from backtester import compare_strategies, print_strategy_comparison

    # Parse arguments
    days = 7
    interval = "1h"

    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        interval = sys.argv[2]

    print("=" * 60)
    print("HISTORICAL DATA FETCHER")
    print("=" * 60)
    print(f"\nFetching {days} days of {interval} candles...")
    print()

    # Fetch data
    candles = fetch_historical_candles(days=days, interval=interval)

    if len(candles["closes"]) == 0:
        print("No data fetched!")
        sys.exit(1)

    # Show stats
    print(f"\nData Summary:")
    print(f"  Period: {candles['timestamps'][0]} to {candles['timestamps'][-1]}")
    print(f"  Candles: {len(candles['closes'])}")
    print(f"  Price Range: ${candles['closes'].min():,.2f} - ${candles['closes'].max():,.2f}")
    price_change = ((candles['closes'][-1] / candles['closes'][0]) - 1) * 100
    print(f"  Price Change: {price_change:+.2f}%")

    # Save data
    save_historical_data(candles, f"historical_{days}d_{interval}.json")

    # Run backtest
    print("\n" + "=" * 60)
    print("RUNNING BACKTEST ON HISTORICAL DATA")
    print("=" * 60 + "\n")

    results = compare_strategies(candles)
    print_strategy_comparison(results)
