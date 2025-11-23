"""
Simple trading strategies without ML.
Each strategy implements a common interface for easy comparison.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple
from indicators import (
    calculate_ema, calculate_sma, calculate_bollinger_bands,
    calculate_rsi, calculate_atr, calculate_support_resistance
)


@dataclass
class StrategySignal:
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class Strategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        """Generate a trading signal from price data."""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}('{self.name}')"


# =============================================================================
# MEAN REVERSION STRATEGY
# =============================================================================
class MeanReversionStrategy(Strategy):
    """
    Mean Reversion: Price tends to return to its average.

    Logic:
    - BUY when price is significantly below moving average (oversold)
    - SELL when price is significantly above moving average (overbought)

    Uses z-score to measure deviation from mean.
    """

    def __init__(self, lookback: int = 20, entry_zscore: float = 2.0,
                 exit_zscore: float = 0.5):
        super().__init__("Mean Reversion")
        self.lookback = lookback
        self.entry_zscore = entry_zscore  # Enter when |z| > this
        self.exit_zscore = exit_zscore    # Exit when |z| < this

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        if len(closes) < self.lookback:
            return StrategySignal("HOLD", 0.0, "Not enough data")

        # Calculate z-score
        window = closes[-self.lookback:]
        mean = np.mean(window)
        std = np.std(window)

        if std == 0:
            return StrategySignal("HOLD", 0.0, "No price variance")

        current_price = closes[-1]
        zscore = (current_price - mean) / std

        # Calculate stop loss / take profit based on ATR
        atr = self._calculate_atr(highs, lows, closes)
        stop_distance = atr * 2 if atr > 0 else current_price * 0.02

        if zscore < -self.entry_zscore:
            # Price is far below mean - BUY expecting reversion up
            confidence = min(abs(zscore) / 3.0, 1.0)
            return StrategySignal(
                "BUY", confidence,
                f"Z-score: {zscore:.2f} (below -{self.entry_zscore})",
                stop_loss=current_price - stop_distance,
                take_profit=mean  # Target the mean
            )

        elif zscore > self.entry_zscore:
            # Price is far above mean - SELL expecting reversion down
            confidence = min(abs(zscore) / 3.0, 1.0)
            return StrategySignal(
                "SELL", confidence,
                f"Z-score: {zscore:.2f} (above +{self.entry_zscore})",
                stop_loss=current_price + stop_distance,
                take_profit=mean
            )

        elif abs(zscore) < self.exit_zscore:
            return StrategySignal("HOLD", 0.5, f"Z-score: {zscore:.2f} (near mean)")

        else:
            return StrategySignal("HOLD", 0.3, f"Z-score: {zscore:.2f} (waiting)")

    def _calculate_atr(self, highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return 0
        atr_vals = calculate_atr(highs[-period-1:], lows[-period-1:], closes[-period-1:], period)
        return atr_vals[-1] if len(atr_vals) > 0 else 0


# =============================================================================
# MOMENTUM STRATEGY
# =============================================================================
class MomentumStrategy(Strategy):
    """
    Momentum: Price tends to continue in its current direction.

    Logic:
    - BUY when price shows strong upward momentum
    - SELL when price shows strong downward momentum

    Uses rate of change (ROC) and EMA crossovers.
    """

    def __init__(self, fast_period: int = 10, slow_period: int = 30,
                 roc_period: int = 10, roc_threshold: float = 2.0):
        super().__init__("Momentum")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold  # % change threshold

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        if len(closes) < self.slow_period + 1:
            return StrategySignal("HOLD", 0.0, "Not enough data")

        current_price = closes[-1]

        # Calculate EMAs
        ema_fast = calculate_ema(closes, self.fast_period)
        ema_slow = calculate_ema(closes, self.slow_period)

        # Calculate Rate of Change (ROC)
        roc = ((current_price - closes[-self.roc_period]) / closes[-self.roc_period]) * 100

        # EMA crossover detection
        ema_diff = ema_fast[-1] - ema_slow[-1]
        ema_diff_prev = ema_fast[-2] - ema_slow[-2]

        # Bullish: fast crosses above slow + positive ROC
        if ema_diff > 0 and ema_diff_prev <= 0 and roc > self.roc_threshold:
            confidence = min(roc / 5.0, 1.0)
            return StrategySignal(
                "BUY", confidence,
                f"Bullish crossover + ROC: {roc:.2f}%",
                stop_loss=current_price * 0.98,
                take_profit=current_price * 1.04
            )

        # Bearish: fast crosses below slow + negative ROC
        elif ema_diff < 0 and ema_diff_prev >= 0 and roc < -self.roc_threshold:
            confidence = min(abs(roc) / 5.0, 1.0)
            return StrategySignal(
                "SELL", confidence,
                f"Bearish crossover + ROC: {roc:.2f}%",
                stop_loss=current_price * 1.02,
                take_profit=current_price * 0.96
            )

        # Strong momentum continuation
        elif ema_diff > 0 and roc > self.roc_threshold * 1.5:
            return StrategySignal("BUY", 0.6, f"Strong upward momentum: ROC {roc:.2f}%")

        elif ema_diff < 0 and roc < -self.roc_threshold * 1.5:
            return StrategySignal("SELL", 0.6, f"Strong downward momentum: ROC {roc:.2f}%")

        else:
            trend = "Bullish" if ema_diff > 0 else "Bearish"
            return StrategySignal("HOLD", 0.3, f"{trend} trend, ROC: {roc:.2f}%")


# =============================================================================
# BREAKOUT STRATEGY
# =============================================================================
class BreakoutStrategy(Strategy):
    """
    Breakout: Trade when price breaks through support/resistance.

    Logic:
    - BUY when price breaks above resistance
    - SELL when price breaks below support

    Uses recent highs/lows as support/resistance levels.
    """

    def __init__(self, lookback: int = 20, confirmation_candles: int = 2,
                 breakout_pct: float = 0.1):
        super().__init__("Breakout")
        self.lookback = lookback
        self.confirmation_candles = confirmation_candles
        self.breakout_pct = breakout_pct  # % above/below level to confirm

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        if len(closes) < self.lookback + self.confirmation_candles:
            return StrategySignal("HOLD", 0.0, "Not enough data")

        current_price = closes[-1]

        # Calculate support/resistance from lookback period (excluding recent candles)
        lookback_highs = highs[-(self.lookback + self.confirmation_candles):-self.confirmation_candles]
        lookback_lows = lows[-(self.lookback + self.confirmation_candles):-self.confirmation_candles]

        resistance = np.max(lookback_highs)
        support = np.min(lookback_lows)

        # Calculate breakout thresholds
        resistance_breakout = resistance * (1 + self.breakout_pct / 100)
        support_breakout = support * (1 - self.breakout_pct / 100)

        # Check for breakout confirmation (price above/below for N candles)
        recent_closes = closes[-self.confirmation_candles:]

        # Bullish breakout: all recent closes above resistance
        if all(c > resistance for c in recent_closes):
            confidence = min((current_price - resistance) / resistance * 100, 1.0)
            return StrategySignal(
                "BUY", confidence,
                f"Breakout above resistance ${resistance:,.2f}",
                stop_loss=resistance * 0.99,  # Stop just below broken resistance
                take_profit=current_price + (current_price - resistance) * 2  # 2:1 R/R
            )

        # Bearish breakout: all recent closes below support
        elif all(c < support for c in recent_closes):
            confidence = min((support - current_price) / support * 100, 1.0)
            return StrategySignal(
                "SELL", confidence,
                f"Breakdown below support ${support:,.2f}",
                stop_loss=support * 1.01,
                take_profit=current_price - (support - current_price) * 2
            )

        # Near resistance (potential breakout)
        elif current_price > resistance * 0.995:
            return StrategySignal("HOLD", 0.4, f"Testing resistance ${resistance:,.2f}")

        # Near support (potential breakdown)
        elif current_price < support * 1.005:
            return StrategySignal("HOLD", 0.4, f"Testing support ${support:,.2f}")

        else:
            return StrategySignal(
                "HOLD", 0.2,
                f"Range: ${support:,.2f} - ${resistance:,.2f}"
            )


# =============================================================================
# GRID TRADING STRATEGY
# =============================================================================
class GridStrategy(Strategy):
    """
    Grid Trading: Place buy/sell orders at fixed price intervals.

    Logic:
    - Divide price range into grid levels
    - BUY when price drops to a grid level
    - SELL when price rises to a grid level

    Good for ranging/sideways markets.
    """

    def __init__(self, grid_size_pct: float = 1.0, num_grids: int = 5):
        super().__init__("Grid Trading")
        self.grid_size_pct = grid_size_pct  # % between grid levels
        self.num_grids = num_grids
        self.last_grid_level = None

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        if len(closes) < 2:
            return StrategySignal("HOLD", 0.0, "Not enough data")

        current_price = closes[-1]
        prev_price = closes[-2]

        # Calculate grid levels based on current price
        grid_step = current_price * (self.grid_size_pct / 100)

        # Round price to nearest grid level
        current_grid = round(current_price / grid_step) * grid_step
        prev_grid = round(prev_price / grid_step) * grid_step

        # Detect grid level crossing
        if current_grid != prev_grid:
            if current_price < prev_price:
                # Price dropped to lower grid - BUY
                return StrategySignal(
                    "BUY", 0.7,
                    f"Grid level: ${current_grid:,.2f} (down)",
                    stop_loss=current_grid - grid_step * 2,
                    take_profit=current_grid + grid_step
                )
            else:
                # Price rose to higher grid - SELL
                return StrategySignal(
                    "SELL", 0.7,
                    f"Grid level: ${current_grid:,.2f} (up)",
                    stop_loss=current_grid + grid_step * 2,
                    take_profit=current_grid - grid_step
                )

        return StrategySignal(
            "HOLD", 0.3,
            f"Between grids: ${current_grid - grid_step:,.0f} - ${current_grid + grid_step:,.0f}"
        )


# =============================================================================
# COMBINED STRATEGY (Multi-Strategy Voting)
# =============================================================================
class CombinedStrategy(Strategy):
    """
    Combines multiple strategies using weighted voting.
    """

    def __init__(self, strategies: List[Tuple[Strategy, float]] = None):
        """
        Args:
            strategies: List of (Strategy, weight) tuples
        """
        super().__init__("Combined")
        if strategies is None:
            strategies = [
                (MeanReversionStrategy(), 1.0),
                (MomentumStrategy(), 1.0),
                (BreakoutStrategy(), 1.0),
            ]
        self.strategies = strategies

    def generate_signal(self, highs: np.ndarray, lows: np.ndarray,
                        closes: np.ndarray) -> StrategySignal:
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        reasons = []

        total_weight = sum(w for _, w in self.strategies)

        for strategy, weight in self.strategies:
            signal = strategy.generate_signal(highs, lows, closes)
            normalized_weight = weight / total_weight

            if signal.action == "BUY":
                buy_score += signal.confidence * normalized_weight
                reasons.append(f"{strategy.name}: BUY ({signal.confidence:.0%})")
            elif signal.action == "SELL":
                sell_score += signal.confidence * normalized_weight
                reasons.append(f"{strategy.name}: SELL ({signal.confidence:.0%})")
            else:
                hold_score += signal.confidence * normalized_weight
                reasons.append(f"{strategy.name}: HOLD")

        # Determine final action
        max_score = max(buy_score, sell_score, hold_score)

        if buy_score == max_score and buy_score > 0.3:
            return StrategySignal("BUY", buy_score, " | ".join(reasons))
        elif sell_score == max_score and sell_score > 0.3:
            return StrategySignal("SELL", sell_score, " | ".join(reasons))
        else:
            return StrategySignal("HOLD", hold_score, " | ".join(reasons))


# =============================================================================
# STRATEGY FACTORY
# =============================================================================
def get_strategy(name: str, **kwargs) -> Strategy:
    """Factory function to create strategies by name."""
    strategies = {
        "mean_reversion": MeanReversionStrategy,
        "momentum": MomentumStrategy,
        "breakout": BreakoutStrategy,
        "grid": GridStrategy,
        "combined": CombinedStrategy,
    }

    if name.lower() not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(strategies.keys())}")

    return strategies[name.lower()](**kwargs)


def list_strategies() -> List[str]:
    """List available strategy names."""
    return ["mean_reversion", "momentum", "breakout", "grid", "combined"]


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    # Demo with random price data
    np.random.seed(42)

    # Generate sample price data
    n = 100
    base_price = 50000
    returns = np.random.randn(n) * 0.01
    closes = base_price * np.cumprod(1 + returns)
    highs = closes * (1 + np.abs(np.random.randn(n) * 0.005))
    lows = closes * (1 - np.abs(np.random.randn(n) * 0.005))

    print("=" * 60)
    print("STRATEGY DEMO")
    print("=" * 60)

    for name in list_strategies():
        strategy = get_strategy(name)
        signal = strategy.generate_signal(highs, lows, closes)

        print(f"\n{strategy.name}:")
        print(f"  Action: {signal.action}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Reason: {signal.reason}")
        if signal.stop_loss:
            print(f"  Stop Loss: ${signal.stop_loss:,.2f}")
        if signal.take_profit:
            print(f"  Take Profit: ${signal.take_profit:,.2f}")
