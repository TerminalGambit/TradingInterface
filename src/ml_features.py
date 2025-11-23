"""
Feature engineering for ML price prediction.
Converts OHLCV data and technical indicators into ML-ready features.
"""

import numpy as np
from typing import Dict, Tuple, List
from indicators import (
    calculate_ema, calculate_sma, calculate_rsi, calculate_macd,
    calculate_bollinger_bands, calculate_atr, calculate_stochastic
)


def create_features(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    volumes: np.ndarray = None, lookback: int = 20) -> Dict[str, np.ndarray]:
    """
    Create feature matrix from OHLCV data.

    Features include:
    - Price-based: returns, volatility, price position
    - Trend indicators: EMA crossovers, MACD
    - Momentum indicators: RSI, Stochastic
    - Volatility indicators: ATR, Bollinger Band width
    - Volume indicators (if available)

    Args:
        highs: High prices
        lows: Low prices
        closes: Close prices
        volumes: Volume data (optional)
        lookback: Lookback period for calculations

    Returns:
        Dict of feature name -> feature values array
    """
    n = len(closes)
    features = {}

    # ==========================================================================
    # PRICE-BASED FEATURES
    # ==========================================================================

    # Returns at different horizons
    features["return_1"] = np.zeros(n)
    features["return_5"] = np.zeros(n)
    features["return_10"] = np.zeros(n)
    features["return_20"] = np.zeros(n)

    for i in range(1, n):
        features["return_1"][i] = (closes[i] / closes[i-1] - 1) * 100

    for i in range(5, n):
        features["return_5"][i] = (closes[i] / closes[i-5] - 1) * 100

    for i in range(10, n):
        features["return_10"][i] = (closes[i] / closes[i-10] - 1) * 100

    for i in range(20, n):
        features["return_20"][i] = (closes[i] / closes[i-20] - 1) * 100

    # Volatility (rolling std of returns)
    returns = np.diff(closes) / closes[:-1]
    features["volatility_10"] = np.zeros(n)
    features["volatility_20"] = np.zeros(n)

    for i in range(11, n):
        features["volatility_10"][i] = np.std(returns[i-10:i]) * 100

    for i in range(21, n):
        features["volatility_20"][i] = np.std(returns[i-20:i]) * 100

    # Price position relative to range
    features["price_position"] = np.zeros(n)
    for i in range(lookback, n):
        high_max = np.max(highs[i-lookback:i+1])
        low_min = np.min(lows[i-lookback:i+1])
        if high_max > low_min:
            features["price_position"][i] = (closes[i] - low_min) / (high_max - low_min)

    # ==========================================================================
    # TREND INDICATORS
    # ==========================================================================

    # EMA crossovers
    ema_10 = calculate_ema(closes, 10)
    ema_20 = calculate_ema(closes, 20)
    ema_50 = calculate_ema(closes, 50) if n > 50 else np.zeros(n)

    features["ema_10_20_diff"] = (ema_10 - ema_20) / closes * 100
    features["ema_10_50_diff"] = np.zeros(n)
    if n > 50:
        features["ema_10_50_diff"] = (ema_10 - ema_50) / closes * 100

    # Price relative to EMAs
    features["price_vs_ema_10"] = (closes - ema_10) / closes * 100
    features["price_vs_ema_20"] = (closes - ema_20) / closes * 100

    # MACD
    if n > 35:
        macd_data = calculate_macd(closes)
        macd_line = macd_data["macd"]
        signal_line = macd_data["signal"]
        histogram = macd_data["histogram"]

        # Pad to match length
        pad = n - len(macd_line)
        features["macd"] = np.concatenate([np.zeros(pad), macd_line])
        features["macd_signal"] = np.concatenate([np.zeros(pad), signal_line])
        features["macd_histogram"] = np.concatenate([np.zeros(pad), histogram])
    else:
        features["macd"] = np.zeros(n)
        features["macd_signal"] = np.zeros(n)
        features["macd_histogram"] = np.zeros(n)

    # ==========================================================================
    # MOMENTUM INDICATORS
    # ==========================================================================

    # RSI
    if n > 15:
        rsi = calculate_rsi(closes, 14)
        pad = n - len(rsi)
        features["rsi"] = np.concatenate([np.full(pad, 50), rsi])
    else:
        features["rsi"] = np.full(n, 50)

    # RSI normalized (0-1 instead of 0-100)
    features["rsi_norm"] = features["rsi"] / 100

    # RSI overbought/oversold signals
    features["rsi_oversold"] = (features["rsi"] < 30).astype(float)
    features["rsi_overbought"] = (features["rsi"] > 70).astype(float)

    # Stochastic
    if n > 17:
        stoch = calculate_stochastic(highs, lows, closes)
        k = stoch["k"]
        d = stoch["d"]

        pad_k = n - len(k)
        pad_d = n - len(d)

        features["stoch_k"] = np.concatenate([np.full(pad_k, 50), k])
        features["stoch_d"] = np.concatenate([np.full(pad_d, 50), d]) if len(d) > 0 else np.full(n, 50)
    else:
        features["stoch_k"] = np.full(n, 50)
        features["stoch_d"] = np.full(n, 50)

    # ==========================================================================
    # VOLATILITY INDICATORS
    # ==========================================================================

    # ATR
    if n > 15:
        atr = calculate_atr(highs, lows, closes, 14)
        pad = n - len(atr)
        features["atr"] = np.concatenate([np.zeros(pad), atr])
        features["atr_pct"] = features["atr"] / closes * 100
    else:
        features["atr"] = np.zeros(n)
        features["atr_pct"] = np.zeros(n)

    # Bollinger Bands
    if n > 21:
        bb = calculate_bollinger_bands(closes, 20, 2)
        upper = bb["upper"]
        lower = bb["lower"]
        middle = bb["middle"]

        pad = n - len(upper)

        # BB width (volatility measure)
        bb_width = (upper - lower) / middle * 100
        features["bb_width"] = np.concatenate([np.zeros(pad), bb_width])

        # BB position (where price is within bands)
        bb_position = np.zeros(len(upper))
        for i in range(len(upper)):
            if upper[i] > lower[i]:
                bb_position[i] = (closes[i + pad] - lower[i]) / (upper[i] - lower[i])
        features["bb_position"] = np.concatenate([np.full(pad, 0.5), bb_position])
    else:
        features["bb_width"] = np.zeros(n)
        features["bb_position"] = np.full(n, 0.5)

    # ==========================================================================
    # VOLUME FEATURES (if available)
    # ==========================================================================

    if volumes is not None and len(volumes) == n:
        # Volume change
        features["volume_change"] = np.zeros(n)
        for i in range(1, n):
            if volumes[i-1] > 0:
                features["volume_change"][i] = (volumes[i] / volumes[i-1] - 1) * 100

        # Volume moving average ratio
        vol_ma = calculate_sma(volumes, 20)
        features["volume_vs_ma"] = np.zeros(n)
        for i in range(20, min(n, len(vol_ma))):
            if vol_ma[i] > 0:
                features["volume_vs_ma"][i] = volumes[i] / vol_ma[i]

        # Price-volume correlation (simple)
        features["price_volume_corr"] = np.zeros(n)
        for i in range(20, n):
            price_changes = np.diff(closes[i-20:i+1])
            vol_changes = np.diff(volumes[i-20:i+1])
            if np.std(price_changes) > 0 and np.std(vol_changes) > 0:
                features["price_volume_corr"][i] = np.corrcoef(price_changes, vol_changes)[0, 1]

    return features


def create_labels(closes: np.ndarray, horizon: int = 1,
                  threshold: float = 0.0) -> np.ndarray:
    """
    Create classification labels for price direction prediction.

    Args:
        closes: Close prices
        horizon: Number of periods ahead to predict
        threshold: Minimum % change to count as up/down (0 = any change)

    Returns:
        Labels: 1 (up), 0 (down), -1 (neutral/unknown for last `horizon` rows)
    """
    n = len(closes)
    labels = np.full(n, -1)  # -1 = unknown (last rows)

    for i in range(n - horizon):
        future_return = (closes[i + horizon] / closes[i] - 1) * 100

        if future_return > threshold:
            labels[i] = 1  # Price goes up
        elif future_return < -threshold:
            labels[i] = 0  # Price goes down
        else:
            labels[i] = -1  # Neutral (within threshold)

    return labels


def create_regression_labels(closes: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Create regression labels (future return %).

    Args:
        closes: Close prices
        horizon: Number of periods ahead to predict

    Returns:
        Future return percentages (NaN for last `horizon` rows)
    """
    n = len(closes)
    labels = np.full(n, np.nan)

    for i in range(n - horizon):
        labels[i] = (closes[i + horizon] / closes[i] - 1) * 100

    return labels


def build_dataset(candles: dict, horizon: int = 1,
                  threshold: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build complete ML dataset from candle data.

    Args:
        candles: Dict with 'opens', 'highs', 'lows', 'closes', 'volumes', 'timestamps'
        horizon: Prediction horizon (candles ahead)
        threshold: Minimum % change for classification

    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Labels (n_samples,)
        feature_names: List of feature names
    """
    highs = candles["highs"]
    lows = candles["lows"]
    closes = candles["closes"]
    volumes = candles.get("volumes")

    # Create features
    features = create_features(highs, lows, closes, volumes)

    # Create labels
    labels = create_labels(closes, horizon, threshold)

    # Convert to matrix
    feature_names = list(features.keys())
    X = np.column_stack([features[name] for name in feature_names])
    y = labels

    # Remove rows with unknown labels
    valid_mask = (y != -1)

    # Also remove first 50 rows (need history for indicators)
    valid_mask[:50] = False

    return X[valid_mask], y[valid_mask], feature_names


def get_feature_importance_names() -> List[str]:
    """Return list of most important features for display."""
    return [
        "return_1", "return_5", "rsi", "macd_histogram",
        "bb_position", "price_vs_ema_20", "volatility_10", "atr_pct"
    ]


if __name__ == "__main__":
    # Demo with historical data
    from fetch_historical import fetch_historical_candles

    print("=" * 60)
    print("ML FEATURE ENGINEERING DEMO")
    print("=" * 60)

    # Fetch data
    print("\nFetching 30 days of hourly data...")
    candles = fetch_historical_candles(days=30, interval="1h")

    if len(candles["closes"]) == 0:
        print("No data fetched!")
        exit(1)

    # Build dataset
    print("\nBuilding ML dataset...")
    X, y, feature_names = build_dataset(candles, horizon=1, threshold=0.1)

    print(f"\nDataset Shape:")
    print(f"  Features (X): {X.shape}")
    print(f"  Labels (y):   {y.shape}")
    print(f"  Feature count: {len(feature_names)}")

    print(f"\nFeatures:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1:2d}. {name}")

    print(f"\nLabel Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for val, count in zip(unique, counts):
        label = "UP" if val == 1 else "DOWN"
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")

    # Show sample
    print(f"\nSample Features (first row):")
    for i, name in enumerate(feature_names[:10]):
        print(f"  {name}: {X[0, i]:.4f}")
