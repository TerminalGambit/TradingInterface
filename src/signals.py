"""
Trading signals based on technical indicators.
Each signal returns: 1 (BUY), -1 (SELL), 0 (HOLD)
"""

import numpy as np
from indicators import (
    calculate_rsi, calculate_macd, calculate_bollinger_bands,
    calculate_ema, calculate_stochastic, calculate_support_resistance
)


def signal_rsi(closes: np.ndarray, period: int = 14,
               oversold: float = 30, overbought: float = 70) -> np.ndarray:
    """
    RSI Signal:
    - BUY when RSI crosses above oversold (30)
    - SELL when RSI crosses below overbought (70)
    """
    if len(closes) < period + 2:
        return np.zeros(len(closes))

    rsi = calculate_rsi(closes, period)

    # Pad to match closes length
    pad = len(closes) - len(rsi)
    signals = np.zeros(len(closes))

    for i in range(1, len(rsi)):
        idx = i + pad
        # Buy: RSI crosses up through oversold
        if rsi[i] > oversold and rsi[i-1] <= oversold:
            signals[idx] = 1
        # Sell: RSI crosses down through overbought
        elif rsi[i] < overbought and rsi[i-1] >= overbought:
            signals[idx] = -1

    return signals


def signal_macd(closes: np.ndarray, fast: int = 12, slow: int = 26,
                signal_period: int = 9) -> np.ndarray:
    """
    MACD Signal:
    - BUY when MACD crosses above signal line
    - SELL when MACD crosses below signal line
    """
    if len(closes) < slow + signal_period:
        return np.zeros(len(closes))

    macd_data = calculate_macd(closes, fast, slow, signal_period)
    macd_line = macd_data["macd"]
    signal_line = macd_data["signal"]

    signals = np.zeros(len(closes))

    for i in range(1, len(macd_line)):
        # Buy: MACD crosses above signal
        if macd_line[i] > signal_line[i] and macd_line[i-1] <= signal_line[i-1]:
            signals[i] = 1
        # Sell: MACD crosses below signal
        elif macd_line[i] < signal_line[i] and macd_line[i-1] >= signal_line[i-1]:
            signals[i] = -1

    return signals


def signal_bollinger(closes: np.ndarray, window: int = 20, num_std: int = 2) -> np.ndarray:
    """
    Bollinger Bands Signal:
    - BUY when price touches/crosses below lower band (oversold)
    - SELL when price touches/crosses above upper band (overbought)
    """
    if len(closes) < window + 1:
        return np.zeros(len(closes))

    bands = calculate_bollinger_bands(closes, window, num_std)
    pad = len(closes) - len(bands["middle"])

    signals = np.zeros(len(closes))

    for i in range(len(bands["lower"])):
        idx = i + pad
        price = closes[idx]

        # Buy: price at or below lower band
        if price <= bands["lower"][i]:
            signals[idx] = 1
        # Sell: price at or above upper band
        elif price >= bands["upper"][i]:
            signals[idx] = -1

    return signals


def signal_ema_crossover(closes: np.ndarray, fast: int = 10, slow: int = 20) -> np.ndarray:
    """
    EMA Crossover Signal:
    - BUY when fast EMA crosses above slow EMA
    - SELL when fast EMA crosses below slow EMA
    """
    if len(closes) < slow + 1:
        return np.zeros(len(closes))

    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)

    signals = np.zeros(len(closes))

    for i in range(1, len(closes)):
        # Buy: fast crosses above slow
        if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]:
            signals[i] = 1
        # Sell: fast crosses below slow
        elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]:
            signals[i] = -1

    return signals


def signal_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                      k_period: int = 14, d_period: int = 3,
                      oversold: float = 20, overbought: float = 80) -> np.ndarray:
    """
    Stochastic Oscillator Signal:
    - BUY when %K crosses above %D in oversold zone
    - SELL when %K crosses below %D in overbought zone
    """
    if len(closes) < k_period + d_period:
        return np.zeros(len(closes))

    stoch = calculate_stochastic(highs, lows, closes, k_period, d_period)
    k = stoch["k"]
    d = stoch["d"]

    if len(k) == 0 or len(d) == 0:
        return np.zeros(len(closes))

    pad_k = len(closes) - len(k)
    pad_d = len(k) - len(d)

    signals = np.zeros(len(closes))

    for i in range(1, len(d)):
        k_idx = i + pad_d
        close_idx = k_idx + pad_k

        if k_idx >= len(k) or k_idx - 1 < 0:
            continue

        # Buy: %K crosses above %D in oversold zone
        if (k[k_idx] > d[i] and k[k_idx-1] <= d[i-1] and d[i] < oversold):
            signals[close_idx] = 1
        # Sell: %K crosses below %D in overbought zone
        elif (k[k_idx] < d[i] and k[k_idx-1] >= d[i-1] and d[i] > overbought):
            signals[close_idx] = -1

    return signals


def combine_signals(signals_list: list, method: str = "majority") -> np.ndarray:
    """
    Combine multiple signals.

    Methods:
    - "majority": Signal when majority agree (>50%)
    - "unanimous": Signal only when all agree
    - "any": Signal when any indicator signals
    - "weighted": Weighted average (pass weights as second element in tuple)
    """
    if not signals_list:
        return np.array([])

    # Stack all signals
    stacked = np.vstack(signals_list)
    n_signals = len(signals_list)

    if method == "majority":
        # Count buy/sell signals at each point
        buys = np.sum(stacked == 1, axis=0)
        sells = np.sum(stacked == -1, axis=0)

        result = np.zeros(stacked.shape[1])
        result[buys > n_signals / 2] = 1
        result[sells > n_signals / 2] = -1
        return result

    elif method == "unanimous":
        result = np.zeros(stacked.shape[1])
        result[np.all(stacked == 1, axis=0)] = 1
        result[np.all(stacked == -1, axis=0)] = -1
        return result

    elif method == "any":
        result = np.zeros(stacked.shape[1])
        result[np.any(stacked == 1, axis=0)] = 1
        result[np.any(stacked == -1, axis=0)] = -1
        # If both buy and sell, hold
        result[(np.any(stacked == 1, axis=0)) & (np.any(stacked == -1, axis=0))] = 0
        return result

    else:
        raise ValueError(f"Unknown method: {method}")


class SignalGenerator:
    """
    Generates trading signals from price data using multiple strategies.
    """

    def __init__(self, config: dict = None):
        self.config = config or {
            "rsi_enabled": True,
            "macd_enabled": True,
            "bollinger_enabled": True,
            "ema_crossover_enabled": True,
            "stochastic_enabled": False,
            "combine_method": "majority",  # majority, unanimous, any
        }

    def generate(self, highs: np.ndarray, lows: np.ndarray,
                 closes: np.ndarray) -> dict:
        """
        Generate signals from price data.
        Returns dict with individual signals and combined signal.
        """
        signals = {}

        if self.config.get("rsi_enabled"):
            signals["rsi"] = signal_rsi(closes)

        if self.config.get("macd_enabled"):
            signals["macd"] = signal_macd(closes)

        if self.config.get("bollinger_enabled"):
            signals["bollinger"] = signal_bollinger(closes)

        if self.config.get("ema_crossover_enabled"):
            signals["ema_crossover"] = signal_ema_crossover(closes)

        if self.config.get("stochastic_enabled"):
            signals["stochastic"] = signal_stochastic(highs, lows, closes)

        # Combine signals
        if signals:
            signals["combined"] = combine_signals(
                list(signals.values()),
                method=self.config.get("combine_method", "majority")
            )
        else:
            signals["combined"] = np.zeros(len(closes))

        return signals

    def get_current_signal(self, highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray) -> dict:
        """Get the most recent signal."""
        signals = self.generate(highs, lows, closes)

        result = {
            "combined": int(signals["combined"][-1]) if len(signals["combined"]) > 0 else 0,
            "details": {}
        }

        for name, sig in signals.items():
            if name != "combined" and len(sig) > 0:
                result["details"][name] = int(sig[-1])

        # Interpret combined signal
        if result["combined"] == 1:
            result["action"] = "BUY"
        elif result["combined"] == -1:
            result["action"] = "SELL"
        else:
            result["action"] = "HOLD"

        return result
