"""
Technical indicators for financial analysis.
"""

import numpy as np


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    multiplier = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    return ema


def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    return np.convolve(prices, np.ones(period), mode='valid') / period


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    Returns dict with 'macd', 'signal', and 'histogram' arrays.
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram
    }


def calculate_bollinger_bands(prices: np.ndarray, window: int = 20, num_std: int = 2) -> dict:
    """
    Calculate Bollinger Bands.
    Returns dict with 'upper', 'middle', and 'lower' arrays.
    """
    middle = np.convolve(prices, np.ones(window), mode='valid') / window
    mean_of_squares = np.convolve(prices**2, np.ones(window), mode='valid') / window
    square_of_mean = middle ** 2
    variance = mean_of_squares - square_of_mean
    rolling_std = np.sqrt(variance)

    upper = middle + (num_std * rolling_std)
    lower = middle - (num_std * rolling_std)

    return {"upper": upper, "middle": middle, "lower": lower}


def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index."""
    price_change = np.diff(prices)
    gains = np.where(price_change > 0, price_change, 0)
    losses = np.where(price_change < 0, -price_change, 0)

    avg_gain = np.convolve(gains, np.ones(period), mode='valid') / period
    avg_loss = np.convolve(losses, np.ones(period), mode='valid') / period

    # Avoid division by zero
    avg_loss = np.where(avg_loss == 0, 1e-10, avg_loss)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR) - volatility indicator.
    Returns ATR values array.
    """
    if len(highs) < 2:
        return np.array([])

    # True Range = max of: (high-low), abs(high-prev_close), abs(low-prev_close)
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]

    tr1 = highs - lows
    tr2 = np.abs(highs - prev_closes)
    tr3 = np.abs(lows - prev_closes)

    true_range = np.maximum(tr1, np.maximum(tr2, tr3))

    # Use EMA for smoothing (Wilder's method)
    atr = calculate_ema(true_range, period)

    return atr


def calculate_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                         k_period: int = 14, d_period: int = 3) -> dict:
    """
    Calculate Stochastic Oscillator.
    Returns dict with 'k' (fast) and 'd' (slow/signal) arrays.
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    """
    if len(closes) < k_period:
        return {"k": np.array([]), "d": np.array([])}

    k_values = []

    for i in range(k_period - 1, len(closes)):
        window_highs = highs[i - k_period + 1:i + 1]
        window_lows = lows[i - k_period + 1:i + 1]

        highest_high = np.max(window_highs)
        lowest_low = np.min(window_lows)

        if highest_high == lowest_low:
            k_values.append(50.0)  # Neutral when no range
        else:
            k = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
            k_values.append(k)

    k_array = np.array(k_values)

    # %D is SMA of %K
    if len(k_array) >= d_period:
        d_array = np.convolve(k_array, np.ones(d_period), mode='valid') / d_period
    else:
        d_array = np.array([])

    return {"k": k_array, "d": d_array}


def calculate_support_resistance(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                                  lookback: int = 20, threshold: float = 0.002) -> dict:
    """
    Calculate support and resistance levels based on recent price action.
    Uses pivot points and price clustering.
    Returns dict with 'support' and 'resistance' levels (list of floats).
    """
    if len(closes) < lookback:
        return {"support": [], "resistance": []}

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]

    current_price = closes[-1]
    price_range = np.max(recent_highs) - np.min(recent_lows)

    if price_range == 0:
        return {"support": [], "resistance": []}

    # Find local maxima (resistance) and minima (support)
    resistance_levels = []
    support_levels = []

    # Method 1: Recent swing highs/lows
    for i in range(2, len(recent_highs) - 2):
        # Local maximum (resistance)
        if (recent_highs[i] > recent_highs[i-1] and
            recent_highs[i] > recent_highs[i-2] and
            recent_highs[i] > recent_highs[i+1] and
            recent_highs[i] > recent_highs[i+2]):
            resistance_levels.append(recent_highs[i])

        # Local minimum (support)
        if (recent_lows[i] < recent_lows[i-1] and
            recent_lows[i] < recent_lows[i-2] and
            recent_lows[i] < recent_lows[i+1] and
            recent_lows[i] < recent_lows[i+2]):
            support_levels.append(recent_lows[i])

    # Method 2: Add absolute recent high/low
    resistance_levels.append(np.max(recent_highs))
    support_levels.append(np.min(recent_lows))

    # Cluster nearby levels (within threshold)
    def cluster_levels(levels, threshold_pct):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] < threshold_pct:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]
        clustered.append(np.mean(current_cluster))
        return clustered

    resistance_levels = cluster_levels(resistance_levels, threshold)
    support_levels = cluster_levels(support_levels, threshold)

    # Filter: resistance above current price, support below
    resistance_levels = [r for r in resistance_levels if r > current_price]
    support_levels = [s for s in support_levels if s < current_price]

    # Take top 3 closest levels
    resistance_levels = sorted(resistance_levels)[:3]
    support_levels = sorted(support_levels, reverse=True)[:3]

    return {"support": support_levels, "resistance": resistance_levels}
