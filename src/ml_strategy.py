"""
ML-based trading strategy.
Uses trained model to generate buy/sell signals.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from strategies import Strategy, StrategySignal
from ml_features import create_features
from ml_model import PriceDirectionModel


class MLStrategy(Strategy):
    """
    Trading strategy based on ML price direction predictions.

    Uses probability thresholds:
    - BUY when P(up) > buy_threshold
    - SELL when P(down) > sell_threshold
    """

    def __init__(self, model_path: str = None,
                 buy_threshold: float = 0.55,
                 sell_threshold: float = 0.55):
        """
        Args:
            model_path: Path to saved model (uses default if None)
            buy_threshold: Min probability to trigger BUY
            sell_threshold: Min probability to trigger SELL
        """
        super().__init__("ML Strategy")
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model = None

        # Load model
        if model_path is None:
            model_path = Path(__file__).parent / "models" / "price_direction_model.pkl"

        if Path(model_path).exists():
            self.model = PriceDirectionModel()
            self.model.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")
            print("Train a model first with: python ml_model.py")

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray, volumes: np.ndarray = None) -> StrategySignal:
        """Generate trading signal from ML prediction."""

        if self.model is None:
            return StrategySignal("HOLD", 0.0, "No model loaded")

        if len(closes) < 50:
            return StrategySignal("HOLD", 0.0, "Not enough data")

        # Create features for latest data point
        features = create_features(highs, lows, closes, volumes)

        # Get last row as feature vector
        feature_names = list(features.keys())
        X = np.array([[features[name][-1] for name in feature_names]])

        # Handle NaN
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)

        # Get prediction probabilities
        try:
            proba = self.model.predict_proba(X)[0]
            prob_down, prob_up = proba[0], proba[1]
        except Exception as e:
            return StrategySignal("HOLD", 0.0, f"Prediction error: {e}")

        # Generate signal based on thresholds
        current_price = closes[-1]

        if prob_up > self.buy_threshold:
            confidence = (prob_up - 0.5) * 2  # Scale 0.5-1.0 to 0-1
            return StrategySignal(
                "BUY", confidence,
                f"ML: {prob_up:.0%} up probability",
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.02
            )

        elif prob_down > self.sell_threshold:
            confidence = (prob_down - 0.5) * 2
            return StrategySignal(
                "SELL", confidence,
                f"ML: {prob_down:.0%} down probability",
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.98
            )

        else:
            return StrategySignal(
                "HOLD", 0.3,
                f"ML: {prob_up:.0%} up, {prob_down:.0%} down (uncertain)"
            )


def backtest_ml_strategy(candles: dict, model_path: str = None,
                          buy_threshold: float = 0.55,
                          sell_threshold: float = 0.55,
                          initial_capital: float = 10000.0):
    """
    Backtest ML strategy and compare with rule-based strategies.
    """
    from backtester import (
        run_strategy_backtest, compare_strategies, print_strategy_comparison,
        print_backtest_report, BacktestResult
    )
    from strategies import MeanReversionStrategy, MomentumStrategy, BreakoutStrategy

    print("=" * 70)
    print("ML STRATEGY BACKTEST")
    print("=" * 70)

    # Create ML strategy
    ml_strategy = MLStrategy(model_path, buy_threshold, sell_threshold)

    if ml_strategy.model is None:
        print("\nError: No model found. Train one first:")
        print("  python ml_model.py")
        return

    # Run ML backtest
    print(f"\nTesting ML Strategy (thresholds: buy={buy_threshold}, sell={sell_threshold})...")

    # Need to pass volumes for ML features
    highs = candles["highs"]
    lows = candles["lows"]
    closes = candles["closes"]
    volumes = candles.get("volumes")
    timestamps = candles["timestamps"]

    # Generate signals
    from backtester import Backtester
    signals = np.zeros(len(closes))

    for i in range(50, len(closes)):
        vol = volumes[:i+1] if volumes is not None else None
        signal = ml_strategy.generate_signal(
            highs[:i+1], lows[:i+1], closes[:i+1], vol
        )
        if signal.action == "BUY":
            signals[i] = 1
        elif signal.action == "SELL":
            signals[i] = -1

    # Run backtest
    backtester = Backtester(initial_capital=initial_capital)
    ml_result = backtester.run(closes, signals, timestamps)

    # Compare with rule-based strategies
    print("\nComparing with rule-based strategies...")
    rule_results = compare_strategies(candles, initial_capital=initial_capital)

    # Add ML result
    all_results = {"ML Strategy": ml_result}
    all_results.update(rule_results)

    # Print comparison
    print_strategy_comparison(all_results)

    # Detailed ML report
    print_backtest_report(ml_result, "ML Strategy")

    return ml_result, rule_results


if __name__ == "__main__":
    from fetch_historical import fetch_historical_candles

    print("Fetching test data (30 days)...")
    candles = fetch_historical_candles(days=30, interval="1h")

    if len(candles["closes"]) == 0:
        print("No data fetched!")
        exit(1)

    # Run backtest
    backtest_ml_strategy(candles)
