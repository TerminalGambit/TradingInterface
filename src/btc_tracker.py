"""
Real-time Bitcoin price tracker with technical indicators.
Fetches BTC/USDT price from Binance API.
Single unified chart with candlesticks and configurable indicators.
Press Ctrl+C to stop.
"""

import requests
import time
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

from indicators import (
    calculate_macd, calculate_ema, calculate_bollinger_bands,
    calculate_rsi, calculate_atr, calculate_stochastic,
    calculate_support_resistance
)


# =============================================================================
# CONFIGURATION - Toggle indicators on/off
# =============================================================================
CONFIG = {
    # Indicators (set to False to disable)
    # Core indicators - most useful for day trading
    "ema": True,
    "bollinger_bands": True,
    "rsi": True,
    "support_resistance": True,

    # Secondary indicators - can be noisy on short timeframes
    "macd": False,              # Better for 1min+ candles, noisy on short timeframes
    "stochastic": False,        # Similar to RSI, pick one to avoid clutter
    "atr": True,                # Useful for volatility/stop-loss sizing

    # API Settings
    "fetch_interval": 1.0,      # 1 request/sec - safe for Binance, less noise
    "candle_period": 15,        # 15s candles - good balance for day trading
    "update_interval": 0.5,     # smoother chart updates

    # Display
    "clear_data_on_start": True,  # Start fresh each session
}

# =============================================================================
# CONSTANTS
# =============================================================================
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"
DATA_FILE = Path(__file__).parent / "btc_prices.json"


def fetch_btc_price() -> dict:
    """Fetch current BTC price from Binance API."""
    response = requests.get(BINANCE_API_URL, params={"symbol": SYMBOL}, timeout=5)
    response.raise_for_status()
    data = response.json()
    return {
        "timestamp": datetime.now().isoformat(),
        "epoch": time.time(),
        "price": float(data["price"])
    }


def load_existing_data() -> list:
    """Load existing price data from file if it exists."""
    if not CONFIG["clear_data_on_start"] and DATA_FILE.exists():
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []


def save_data(data: list) -> None:
    """Save price data to file."""
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)


def aggregate_to_candles(price_data: list) -> list:
    """Aggregate tick data into OHLC candles."""
    if not price_data:
        return []

    candle_period = CONFIG["candle_period"]
    candles = []
    current_candle = None
    candle_start = None

    for tick in price_data:
        epoch = tick.get("epoch", time.time())
        price = tick["price"]

        if candle_start is None:
            candle_start = epoch
            current_candle = {
                "open": price, "high": price, "low": price, "close": price,
                "start_time": epoch
            }
        elif epoch - candle_start >= candle_period:
            candles.append(current_candle)
            candle_start = epoch
            current_candle = {
                "open": price, "high": price, "low": price, "close": price,
                "start_time": epoch
            }
        else:
            current_candle["high"] = max(current_candle["high"], price)
            current_candle["low"] = min(current_candle["low"], price)
            current_candle["close"] = price

    if current_candle:
        candles.append(current_candle)

    return candles


def create_candlestick_collections(candles, min_body_height):
    """Create optimized collections for candlesticks instead of individual patches."""
    green_bodies = []
    red_bodies = []
    green_wicks = []
    red_wicks = []

    width = 0.6

    for i, candle in enumerate(candles):
        is_green = candle["close"] >= candle["open"]

        # Wick line segment
        wick = [(i, candle["low"]), (i, candle["high"])]

        # Body rectangle
        body_bottom = min(candle["open"], candle["close"])
        body_height = abs(candle["close"] - candle["open"])
        if body_height < min_body_height:
            body_height = min_body_height

        rect = Rectangle((i - width/2, body_bottom), width, body_height)

        if is_green:
            green_bodies.append(rect)
            green_wicks.append(wick)
        else:
            red_bodies.append(rect)
            red_wicks.append(wick)

    return {
        "green_bodies": green_bodies,
        "red_bodies": red_bodies,
        "green_wicks": green_wicks,
        "red_wicks": red_wicks
    }


def draw_chart(ax, ax_secondary, candles):
    """Draw unified single chart with all indicators overlaid."""
    ax.clear()
    ax_secondary.clear()

    if len(candles) < 2:
        ax.set_title("BTC/USDT - Waiting for data...")
        ax.set_ylabel("Price (USDT)")
        return

    # Extract OHLC data
    closes = np.array([c["close"] for c in candles])
    highs = np.array([c["high"] for c in candles])
    lows = np.array([c["low"] for c in candles])
    n = len(candles)

    price_range = highs.max() - lows.min()
    min_body_height = price_range * 0.002 if price_range > 0 else 1

    # === CANDLESTICKS (using collections for performance) ===
    candle_data = create_candlestick_collections(candles, min_body_height)

    # Add body collections
    if candle_data["green_bodies"]:
        green_collection = PatchCollection(candle_data["green_bodies"],
                                           facecolor="green", edgecolor="green")
        ax.add_collection(green_collection)

    if candle_data["red_bodies"]:
        red_collection = PatchCollection(candle_data["red_bodies"],
                                         facecolor="red", edgecolor="red")
        ax.add_collection(red_collection)

    # Add wick collections
    if candle_data["green_wicks"]:
        green_wicks = LineCollection(candle_data["green_wicks"], colors="green", linewidths=1)
        ax.add_collection(green_wicks)

    if candle_data["red_wicks"]:
        red_wicks = LineCollection(candle_data["red_wicks"], colors="red", linewidths=1)
        ax.add_collection(red_wicks)

    # === SUPPORT/RESISTANCE LINES ===
    if CONFIG["support_resistance"] and n >= 20:
        sr = calculate_support_resistance(highs, lows, closes, lookback=min(n, 50))

        for level in sr["resistance"]:
            ax.axhline(y=level, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(n + 0.5, level, f"R:{level:,.0f}", fontsize=7, color="red", va="center")

        for level in sr["support"]:
            ax.axhline(y=level, color="green", linestyle="--", linewidth=1, alpha=0.6)
            ax.text(n + 0.5, level, f"S:{level:,.0f}", fontsize=7, color="green", va="center")

    # === BOLLINGER BANDS ===
    if CONFIG["bollinger_bands"] and n >= 20:
        bands = calculate_bollinger_bands(closes, window=20, num_std=2)
        band_x = np.arange(n - len(bands["middle"]), n)
        ax.fill_between(band_x, bands["lower"], bands["upper"], alpha=0.1, color="blue")
        ax.plot(band_x, bands["middle"], "b--", linewidth=1, alpha=0.5, label="BB")

    # === EMA ===
    if CONFIG["ema"] and n >= 10:
        ema10 = calculate_ema(closes, 10)
        ax.plot(range(n), ema10, "orange", linewidth=1.5, alpha=0.7, label="EMA(10)")

    # === SECONDARY AXIS INDICATORS ===
    # We'll stack RSI, Stochastic, and MACD info on the secondary axis
    secondary_labels = []

    # === RSI ===
    if CONFIG["rsi"] and n >= 15:
        rsi = calculate_rsi(closes, period=14)
        current_rsi = rsi[-1] if len(rsi) > 0 else 50
        status = "OB" if current_rsi > 70 else "OS" if current_rsi < 30 else ""
        secondary_labels.append(f"RSI: {current_rsi:.0f} {status}")

    # === STOCHASTIC ===
    if CONFIG["stochastic"] and n >= 14:
        stoch = calculate_stochastic(highs, lows, closes, k_period=14, d_period=3)
        if len(stoch["k"]) > 0:
            current_k = stoch["k"][-1]
            current_d = stoch["d"][-1] if len(stoch["d"]) > 0 else current_k
            stoch_status = "OB" if current_k > 80 else "OS" if current_k < 20 else ""
            secondary_labels.append(f"Stoch: %K={current_k:.0f} %D={current_d:.0f} {stoch_status}")

    # === MACD ===
    if CONFIG["macd"] and n >= 26:
        macd_data = calculate_macd(closes, fast=12, slow=26, signal=9)
        current_macd = macd_data["macd"][-1]
        current_signal = macd_data["signal"][-1]
        macd_status = "BULL" if current_macd > current_signal else "BEAR"
        secondary_labels.append(f"MACD: {current_macd:.1f} Sig: {current_signal:.1f} {macd_status}")

    # === ATR ===
    if CONFIG["atr"] and n >= 14:
        atr = calculate_atr(highs, lows, closes, period=14)
        if len(atr) > 0:
            current_atr = atr[-1]
            atr_pct = (current_atr / closes[-1]) * 100
            secondary_labels.append(f"ATR: {current_atr:.1f} ({atr_pct:.2f}%)")

    # Display indicators as text on secondary axis
    ax_secondary.set_ylim(0, 100)
    ax_secondary.set_yticks([])

    for i, label in enumerate(secondary_labels):
        y_pos = 90 - (i * 15)
        color = "purple" if "RSI" in label else "teal" if "Stoch" in label else \
                "cyan" if "MACD" in label else "gray"
        ax_secondary.text(1.02, y_pos/100, label, transform=ax_secondary.transAxes,
                         fontsize=8, color=color, va="top", fontfamily="monospace")

    # === MAIN AXIS SETUP ===
    ax.set_xlim(-1, n + 5)
    price_margin = price_range * 0.1 if price_range > 0 else closes.mean() * 0.001
    ax.set_ylim(lows.min() - price_margin, highs.max() + price_margin)

    # Title with current price and change
    current_price = closes[-1]
    if n > 1:
        price_change = ((closes[-1] - closes[0]) / closes[0]) * 100
        change_str = f"+{price_change:.2f}%" if price_change >= 0 else f"{price_change:.2f}%"
    else:
        change_str = ""

    ax.set_title(f"BTC/USDT: ${current_price:,.2f} ({change_str}) | {n} candles ({CONFIG['candle_period']}s)",
                fontsize=12, fontweight='bold')
    ax.set_ylabel("Price (USDT)")
    ax.set_xlabel("Candle #")
    ax.grid(True, alpha=0.3)

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper left", fontsize=8)


def print_config():
    """Print current configuration."""
    print("\n=== Configuration ===")
    print(f"Fetch interval: {CONFIG['fetch_interval']}s")
    print(f"Candle period: {CONFIG['candle_period']}s")
    print(f"Clear data on start: {CONFIG['clear_data_on_start']}")
    print("\nIndicators enabled:")
    for key in ["ema", "bollinger_bands", "macd", "rsi", "stochastic", "atr", "support_resistance"]:
        status = "ON" if CONFIG[key] else "OFF"
        print(f"  {key}: {status}")
    print("=" * 25 + "\n")


def run_tracker():
    """Main loop: fetch prices, save to file, update unified chart."""
    print_config()

    price_data = load_existing_data()
    if CONFIG["clear_data_on_start"]:
        print("Starting fresh session (previous data cleared)")

    # Setup single figure with secondary axis for indicator labels
    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 8))
    ax_secondary = ax.twinx()

    plt.tight_layout()
    plt.subplots_adjust(right=0.82)
    plt.show()

    print(f"Data saved to: {DATA_FILE}")
    print("Press Ctrl+C to stop.\n")

    last_update_time = 0
    candles = []  # Cache candles to avoid double computation

    try:
        while True:
            new_data = fetch_btc_price()
            price_data.append(new_data)

            if len(price_data) % 20 == 0:
                save_data(price_data)

            current_time = time.time()

            if current_time - last_update_time >= CONFIG["update_interval"]:
                last_update_time = current_time
                candles = aggregate_to_candles(price_data)  # Compute once

                if candles:
                    draw_chart(ax, ax_secondary, candles)

                fig.canvas.draw()
                fig.canvas.flush_events()

            # Status line (reuse cached candles count)
            print(f"\r[{new_data['timestamp'][11:19]}] BTC: ${new_data['price']:,.2f} | "
                  f"Ticks: {len(price_data)} | Candles: {len(candles)}",
                  end="", flush=True)

            time.sleep(CONFIG["fetch_interval"])

    except KeyboardInterrupt:
        print(f"\n\nStopped. Total ticks: {len(price_data)}")
        save_data(price_data)
        print(f"Data saved to: {DATA_FILE}")
    except Exception as e:
        print(f"\n\nError: {e}")
        save_data(price_data)
        print(f"Data saved to: {DATA_FILE}")
        raise
    finally:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    run_tracker()
