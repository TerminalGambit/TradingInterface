"""
Backtesting engine for trading strategies.
Tests signals against historical data and calculates performance metrics.
Supports both signal-based and strategy-based backtesting.
"""

import json
import gzip
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from signals import SignalGenerator
from strategies import (
    Strategy, get_strategy, list_strategies,
    MeanReversionStrategy, MomentumStrategy, BreakoutStrategy, GridStrategy, CombinedStrategy
)


@dataclass
class Trade:
    entry_time: str
    entry_price: float
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    side: str = "long"  # "long" or "short"
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    total_return: float
    total_return_pct: float
    num_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade]
    equity_curve: np.ndarray
    timestamps: List[str]


class Backtester:
    """
    Backtesting engine for signal-based strategies.
    """

    def __init__(self, initial_capital: float = 10000.0,
                 position_size: float = 1.0,
                 commission: float = 0.001):
        """
        Args:
            initial_capital: Starting capital in USD
            position_size: Fraction of capital to use per trade (0.0-1.0)
            commission: Commission per trade as decimal (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.commission = commission

    def run(self, prices: np.ndarray, signals: np.ndarray,
            timestamps: List[str] = None, allow_short: bool = True) -> BacktestResult:
        """
        Run backtest on price data with given signals.

        Args:
            prices: Array of close prices
            signals: Array of signals (1=buy, -1=sell, 0=hold)
            timestamps: Optional list of timestamps
            allow_short: Allow short selling (SELL first, BUY to cover)

        Returns:
            BacktestResult with performance metrics
        """
        if timestamps is None:
            timestamps = [str(i) for i in range(len(prices))]

        capital = self.initial_capital
        position = 0.0  # BTC held (negative = short)
        entry_price = 0.0
        entry_time = ""
        position_side = None  # "long" or "short"

        trades: List[Trade] = []
        equity_curve = [capital]

        for i in range(len(prices)):
            price = prices[i]
            signal = signals[i]

            # Calculate current equity (for shorts: profit when price drops)
            if position_side == "short":
                unrealized_pnl = (entry_price - price) * abs(position)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital + (position * price)

            # Process signals
            if signal == 1 and position == 0:  # BUY signal, no position
                # Open long position
                trade_capital = capital * self.position_size
                commission_cost = trade_capital * self.commission
                position = (trade_capital - commission_cost) / price
                capital -= trade_capital
                entry_price = price
                entry_time = timestamps[i]
                position_side = "long"

            elif signal == -1 and position > 0:  # SELL signal, have long position
                # Close long position
                sale_value = position * price
                commission_cost = sale_value * self.commission
                capital += sale_value - commission_cost

                # Record trade
                pnl = (price - entry_price) * position - (commission_cost * 2)
                pnl_pct = ((price / entry_price) - 1) * 100

                trades.append(Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=timestamps[i],
                    exit_price=price,
                    side="long",
                    size=position,
                    pnl=pnl,
                    pnl_pct=pnl_pct
                ))

                position = 0.0
                entry_price = 0.0
                position_side = None

            elif signal == -1 and position == 0 and allow_short:  # SELL signal, no position -> short
                # Open short position
                trade_capital = capital * self.position_size
                commission_cost = trade_capital * self.commission
                position = -((trade_capital - commission_cost) / price)  # Negative for short
                entry_price = price
                entry_time = timestamps[i]
                position_side = "short"

            elif signal == 1 and position < 0:  # BUY signal, have short position -> cover
                # Close short position (buy to cover)
                cover_cost = abs(position) * price
                commission_cost = cover_cost * self.commission

                # Short profit = (entry - exit) * size
                pnl = (entry_price - price) * abs(position) - (commission_cost * 2)
                pnl_pct = ((entry_price / price) - 1) * 100

                capital += pnl  # Add/subtract P&L (we didn't actually spend capital on short)

                trades.append(Trade(
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=timestamps[i],
                    exit_price=price,
                    side="short",
                    size=abs(position),
                    pnl=pnl,
                    pnl_pct=pnl_pct
                ))

                position = 0.0
                entry_price = 0.0
                position_side = None

            # Update equity curve
            if position_side == "short":
                unrealized_pnl = (entry_price - price) * abs(position)
                equity_curve.append(capital + unrealized_pnl)
            else:
                equity_curve.append(capital + (position * price))

        # Close any open position at the end
        if position > 0:  # Close long
            final_price = prices[-1]
            sale_value = position * final_price
            commission_cost = sale_value * self.commission
            capital += sale_value - commission_cost

            pnl = (final_price - entry_price) * position - (commission_cost * 2)
            pnl_pct = ((final_price / entry_price) - 1) * 100

            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=timestamps[-1],
                exit_price=final_price,
                side="long",
                size=position,
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

        elif position < 0:  # Close short
            final_price = prices[-1]
            cover_cost = abs(position) * final_price
            commission_cost = cover_cost * self.commission

            pnl = (entry_price - final_price) * abs(position) - (commission_cost * 2)
            pnl_pct = ((entry_price / final_price) - 1) * 100

            capital += pnl

            trades.append(Trade(
                entry_time=entry_time,
                entry_price=entry_price,
                exit_time=timestamps[-1],
                exit_price=final_price,
                side="short",
                size=abs(position),
                pnl=pnl,
                pnl_pct=pnl_pct
            ))

        # Calculate metrics
        equity_curve = np.array(equity_curve)
        return self._calculate_metrics(trades, equity_curve, timestamps)

    def _calculate_metrics(self, trades: List[Trade],
                           equity_curve: np.ndarray,
                           timestamps: List[str]) -> BacktestResult:
        """Calculate performance metrics from trades and equity curve."""

        # Basic stats
        total_return = equity_curve[-1] - self.initial_capital
        total_return_pct = (total_return / self.initial_capital) * 100

        num_trades = len(trades)
        if num_trades == 0:
            return BacktestResult(
                total_return=total_return,
                total_return_pct=total_return_pct,
                num_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=equity_curve,
                timestamps=timestamps
            )

        # Win/loss stats
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]

        win_rate = len(winning_trades) / num_trades * 100 if num_trades > 0 else 0

        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        max_drawdown_pct = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            drawdown_pct = (drawdown / peak) * 100 if peak > 0 else 0

            if drawdown > max_drawdown:
                max_drawdown = drawdown
                max_drawdown_pct = drawdown_pct

        # Sharpe Ratio (assuming daily returns, annualized)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            # Annualize assuming ~252 trading days
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        return BacktestResult(
            total_return=total_return,
            total_return_pct=total_return_pct,
            num_trades=num_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe_ratio,
            trades=trades,
            equity_curve=equity_curve,
            timestamps=timestamps
        )


def load_training_data(data_dir: Path = None) -> dict:
    """
    Load historical data from training_data directory.
    Returns dict with 'prices', 'timestamps', 'highs', 'lows'.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "training_data"

    all_ticks = []

    # Load all JSON and gzipped files
    for filepath in sorted(data_dir.glob("btc_*.json")):
        with open(filepath, "r") as f:
            all_ticks.extend(json.load(f))

    for filepath in sorted(data_dir.glob("btc_*.json.gz")):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            all_ticks.extend(json.load(f))

    if not all_ticks:
        return {"prices": np.array([]), "timestamps": [], "highs": np.array([]), "lows": np.array([])}

    # Sort by epoch
    all_ticks.sort(key=lambda x: x.get("epoch", 0))

    prices = np.array([t["price"] for t in all_ticks])
    timestamps = [t.get("timestamp", "") for t in all_ticks]

    # For tick data, high/low are same as price
    return {
        "prices": prices,
        "timestamps": timestamps,
        "highs": prices.copy(),
        "lows": prices.copy()
    }


def aggregate_to_candles(ticks: list, candle_period: float = 60) -> dict:
    """Aggregate tick data to OHLC candles."""
    if not ticks:
        return {"opens": [], "highs": [], "lows": [], "closes": [], "timestamps": []}

    candles = []
    current = None
    candle_start = None

    for tick in ticks:
        epoch = tick.get("epoch", 0)
        price = tick["price"]

        if candle_start is None:
            candle_start = epoch
            current = {"open": price, "high": price, "low": price, "close": price,
                      "timestamp": tick.get("timestamp", "")}
        elif epoch - candle_start >= candle_period:
            candles.append(current)
            candle_start = epoch
            current = {"open": price, "high": price, "low": price, "close": price,
                      "timestamp": tick.get("timestamp", "")}
        else:
            current["high"] = max(current["high"], price)
            current["low"] = min(current["low"], price)
            current["close"] = price

    if current:
        candles.append(current)

    return {
        "opens": np.array([c["open"] for c in candles]),
        "highs": np.array([c["high"] for c in candles]),
        "lows": np.array([c["low"] for c in candles]),
        "closes": np.array([c["close"] for c in candles]),
        "timestamps": [c["timestamp"] for c in candles]
    }


def run_backtest_on_training_data(signal_config: dict = None,
                                   candle_period: float = 60,
                                   initial_capital: float = 10000.0) -> BacktestResult:
    """
    Run backtest on collected training data.

    Args:
        signal_config: Configuration for SignalGenerator
        candle_period: Candle period in seconds for aggregation
        initial_capital: Starting capital

    Returns:
        BacktestResult with performance metrics
    """
    # Load raw tick data
    data_dir = Path(__file__).parent / "training_data"
    all_ticks = []

    for filepath in sorted(data_dir.glob("btc_*.json")):
        with open(filepath, "r") as f:
            all_ticks.extend(json.load(f))

    for filepath in sorted(data_dir.glob("btc_*.json.gz")):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            all_ticks.extend(json.load(f))

    if not all_ticks:
        print("No training data found!")
        return None

    all_ticks.sort(key=lambda x: x.get("epoch", 0))

    # Aggregate to candles
    candles = aggregate_to_candles(all_ticks, candle_period)

    if len(candles["closes"]) < 30:
        print(f"Not enough data: {len(candles['closes'])} candles (need at least 30)")
        return None

    print(f"Loaded {len(all_ticks)} ticks -> {len(candles['closes'])} candles ({candle_period}s each)")

    # Generate signals
    signal_gen = SignalGenerator(signal_config)
    signals = signal_gen.generate(
        candles["highs"],
        candles["lows"],
        candles["closes"]
    )

    # Run backtest
    backtester = Backtester(initial_capital=initial_capital)
    result = backtester.run(
        candles["closes"],
        signals["combined"],
        candles["timestamps"]
    )

    return result


def run_strategy_backtest(strategy: Strategy, candles: dict,
                          initial_capital: float = 10000.0,
                          commission: float = 0.001) -> BacktestResult:
    """
    Run backtest using a Strategy object instead of signals.

    Args:
        strategy: Strategy instance to test
        candles: Dict with 'highs', 'lows', 'closes', 'timestamps'
        initial_capital: Starting capital
        commission: Commission per trade

    Returns:
        BacktestResult with performance metrics
    """
    highs = candles["highs"]
    lows = candles["lows"]
    closes = candles["closes"]
    timestamps = candles["timestamps"]

    # Convert strategy signals to numeric signals
    signals = np.zeros(len(closes))

    # Generate signal at each point using historical data up to that point
    for i in range(30, len(closes)):  # Need minimum history
        signal = strategy.generate_signal(
            highs[:i+1], lows[:i+1], closes[:i+1]
        )
        if signal.action == "BUY":
            signals[i] = 1
        elif signal.action == "SELL":
            signals[i] = -1

    # Run backtest with generated signals
    backtester = Backtester(initial_capital=initial_capital, commission=commission)
    return backtester.run(closes, signals, timestamps)


def compare_strategies(candles: dict, strategies: List[Strategy] = None,
                       initial_capital: float = 10000.0) -> Dict[str, BacktestResult]:
    """
    Compare multiple strategies on the same data.

    Args:
        candles: Dict with 'highs', 'lows', 'closes', 'timestamps'
        strategies: List of Strategy instances (defaults to all available)
        initial_capital: Starting capital for each strategy

    Returns:
        Dict mapping strategy name to BacktestResult
    """
    if strategies is None:
        strategies = [
            MeanReversionStrategy(),
            MomentumStrategy(),
            BreakoutStrategy(),
            GridStrategy(),
            CombinedStrategy(),
        ]

    results = {}
    for strategy in strategies:
        print(f"  Testing {strategy.name}...", end=" ", flush=True)
        result = run_strategy_backtest(strategy, candles, initial_capital)
        results[strategy.name] = result
        print(f"Return: {result.total_return_pct:+.2f}%")

    return results


def print_strategy_comparison(results: Dict[str, BacktestResult]):
    """Print a comparison table of strategy results."""
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)

    # Header
    print(f"\n{'Strategy':<20} {'Return':>10} {'Trades':>8} {'Win Rate':>10} "
          f"{'Sharpe':>8} {'Max DD':>10}")
    print("-" * 80)

    # Sort by return
    sorted_results = sorted(results.items(), key=lambda x: x[1].total_return_pct, reverse=True)

    for name, result in sorted_results:
        print(f"{name:<20} {result.total_return_pct:>+9.2f}% {result.num_trades:>8} "
              f"{result.win_rate:>9.1f}% {result.sharpe_ratio:>8.2f} "
              f"{result.max_drawdown_pct:>9.2f}%")

    # Best strategy summary
    best_name, best_result = sorted_results[0]
    print("\n" + "-" * 80)
    print(f"BEST STRATEGY: {best_name}")
    print(f"  Return: ${best_result.total_return:.2f} ({best_result.total_return_pct:+.2f}%)")
    print(f"  {best_result.num_trades} trades, {best_result.win_rate:.1f}% win rate")
    print("=" * 80)


def print_backtest_report(result: BacktestResult, strategy_name: str = "Signal-Based"):
    """Print a formatted backtest report."""
    print("\n" + "=" * 60)
    print(f"BACKTEST REPORT: {strategy_name}")
    print("=" * 60)

    print(f"\n📊 PERFORMANCE SUMMARY")
    print(f"  Total Return:      ${result.total_return:,.2f} ({result.total_return_pct:+.2f}%)")
    print(f"  Max Drawdown:      ${result.max_drawdown:,.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f}")

    print(f"\n📈 TRADE STATISTICS")
    print(f"  Total Trades:      {result.num_trades}")
    print(f"  Winning Trades:    {result.winning_trades}")
    print(f"  Losing Trades:     {result.losing_trades}")
    print(f"  Win Rate:          {result.win_rate:.1f}%")

    if result.num_trades > 0:
        print(f"\n💰 PROFIT/LOSS")
        print(f"  Average Win:       ${result.avg_win:,.2f}")
        print(f"  Average Loss:      ${result.avg_loss:,.2f}")
        print(f"  Profit Factor:     {result.profit_factor:.2f}")

        print(f"\n📋 RECENT TRADES")
        for trade in result.trades[-5:]:
            symbol = "🟢" if trade.pnl > 0 else "🔴"
            print(f"  {symbol} {trade.entry_time[:19]} -> {trade.exit_time[:19] if trade.exit_time else 'OPEN'}")
            exit_price = trade.exit_price if trade.exit_price else 0
            print(f"     Entry: ${trade.entry_price:,.2f} | Exit: ${exit_price:,.2f} | P&L: ${trade.pnl:,.2f} ({trade.pnl_pct:+.2f}%)")

    print("\n" + "=" * 60)


def collect_data_for_backtest(min_candles: int = 50, candle_period: float = 60,
                               fetch_interval: float = 1.0):
    """
    Collect enough data for backtesting.

    Args:
        min_candles: Minimum number of candles needed
        candle_period: Seconds per candle
        fetch_interval: Seconds between API calls
    """
    import requests
    import time
    from datetime import datetime

    BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"
    SYMBOL = "BTCUSDT"
    data_dir = Path(__file__).parent / "training_data"
    data_dir.mkdir(exist_ok=True)

    # Calculate how many ticks we need
    ticks_per_candle = candle_period / fetch_interval
    min_ticks = int(min_candles * ticks_per_candle)

    # Check existing data
    all_ticks = []
    for filepath in sorted(data_dir.glob("btc_*.json")):
        with open(filepath, "r") as f:
            all_ticks.extend(json.load(f))

    current_ticks = len(all_ticks)
    ticks_needed = max(0, min_ticks - current_ticks)

    if ticks_needed == 0:
        print(f"Already have enough data: {current_ticks} ticks")
        return

    # Estimate time
    time_needed = ticks_needed * fetch_interval
    minutes = int(time_needed // 60)
    seconds = int(time_needed % 60)

    print(f"\n{'='*60}")
    print("DATA COLLECTION FOR BACKTEST")
    print(f"{'='*60}")
    print(f"Current ticks:  {current_ticks}")
    print(f"Ticks needed:   {min_ticks}")
    print(f"To collect:     {ticks_needed}")
    print(f"Estimated time: {minutes}m {seconds}s")
    print(f"{'='*60}")
    print("\nCollecting data... (Press Ctrl+C to stop early)\n")

    # Get current hour's file
    current_hour = datetime.now().hour
    filepath = data_dir / f"btc_{datetime.now().strftime('%Y_%m_%d_%H')}.json"

    # Load existing data for this hour
    if filepath.exists():
        with open(filepath, "r") as f:
            hourly_data = json.load(f)
    else:
        hourly_data = []

    collected = 0
    try:
        while collected < ticks_needed:
            # Fetch price
            try:
                response = requests.get(BINANCE_API_URL, params={"symbol": SYMBOL}, timeout=5)
                response.raise_for_status()
                price = float(response.json()["price"])

                tick = {
                    "timestamp": datetime.now().isoformat(),
                    "epoch": time.time(),
                    "price": price
                }
                hourly_data.append(tick)
                collected += 1

                # Progress
                progress = (collected / ticks_needed) * 100
                remaining = (ticks_needed - collected) * fetch_interval
                print(f"\r[{progress:5.1f}%] ${price:,.2f} | Collected: {collected}/{ticks_needed} | "
                      f"Remaining: {int(remaining)}s", end="", flush=True)

                # Check for hour rollover
                now = datetime.now()
                if now.hour != current_hour:
                    # Save current hour
                    with open(filepath, "w") as f:
                        json.dump(hourly_data, f)

                    # Start new hour
                    current_hour = now.hour
                    filepath = data_dir / f"btc_{now.strftime('%Y_%m_%d_%H')}.json"
                    hourly_data = []

                # Save periodically
                if collected % 60 == 0:
                    with open(filepath, "w") as f:
                        json.dump(hourly_data, f)

            except requests.RequestException as e:
                print(f"\n[Error] {e}")

            time.sleep(fetch_interval)

    except KeyboardInterrupt:
        print(f"\n\nStopped early. Collected {collected} ticks.")

    # Final save
    with open(filepath, "w") as f:
        json.dump(hourly_data, f)

    print(f"\n\nData collection complete! Saved to {data_dir}")


if __name__ == "__main__":
    import sys

    CANDLE_PERIOD = 60  # 1-minute candles
    MIN_CANDLES = 50    # Need at least 50 candles for good backtest

    # Parse command line args
    mode = "compare"  # Default to strategy comparison
    if len(sys.argv) > 1:
        if sys.argv[1] == "--signals":
            mode = "signals"
        elif sys.argv[1] == "--strategy":
            mode = "single"
        elif sys.argv[1] == "--help":
            print("Usage: python backtester.py [OPTIONS]")
            print("\nOptions:")
            print("  --signals    Run signal-based backtest (RSI, MACD, etc.)")
            print("  --strategy   Run single strategy backtest")
            print("  --compare    Compare all strategies (default)")
            print("\nStrategies: mean_reversion, momentum, breakout, grid, combined")
            sys.exit(0)

    print("=" * 60)
    print("BACKTESTER")
    print("=" * 60)

    # Check if we have enough data
    data_dir = Path(__file__).parent / "training_data"
    all_ticks = []

    for filepath in sorted(data_dir.glob("btc_*.json")):
        with open(filepath, "r") as f:
            all_ticks.extend(json.load(f))

    for filepath in sorted(data_dir.glob("btc_*.json.gz")):
        with gzip.open(filepath, "rt", encoding="utf-8") as f:
            all_ticks.extend(json.load(f))

    # Estimate candle count
    if all_ticks:
        all_ticks.sort(key=lambda x: x.get("epoch", 0))
        time_span = all_ticks[-1].get("epoch", 0) - all_ticks[0].get("epoch", 0)
        estimated_candles = int(time_span / CANDLE_PERIOD) if time_span > 0 else 0
    else:
        estimated_candles = 0

    print(f"\nFound {len(all_ticks)} ticks (~{estimated_candles} candles at {CANDLE_PERIOD}s each)")

    if estimated_candles < MIN_CANDLES:
        print(f"\nNot enough data for backtest (need ~{MIN_CANDLES} candles)")

        response = input("\nCollect data now? [Y/n]: ").strip().lower()
        if response != 'n':
            collect_data_for_backtest(
                min_candles=MIN_CANDLES,
                candle_period=CANDLE_PERIOD,
                fetch_interval=1.0
            )
            # Reload data
            all_ticks = []
            for filepath in sorted(data_dir.glob("btc_*.json")):
                with open(filepath, "r") as f:
                    all_ticks.extend(json.load(f))
            all_ticks.sort(key=lambda x: x.get("epoch", 0))
            print("\nData collection complete. Running backtest...\n")
        else:
            print("Exiting. Run data_collector.py to gather more data.")
            sys.exit(0)

    # Aggregate to candles
    candles = aggregate_to_candles(all_ticks, CANDLE_PERIOD)
    print(f"Aggregated to {len(candles['closes'])} candles\n")

    if mode == "signals":
        # Signal-based backtest
        config = {
            "rsi_enabled": True,
            "macd_enabled": True,
            "bollinger_enabled": True,
            "ema_crossover_enabled": True,
            "stochastic_enabled": False,
            "combine_method": "majority",
        }

        result = run_backtest_on_training_data(
            signal_config=config,
            candle_period=CANDLE_PERIOD,
            initial_capital=10000.0
        )

        if result:
            print_backtest_report(result, "Signal-Based (RSI+MACD+BB+EMA)")

    elif mode == "single":
        # Single strategy backtest
        strategy_name = sys.argv[2] if len(sys.argv) > 2 else "momentum"
        try:
            strategy = get_strategy(strategy_name)
            print(f"Testing strategy: {strategy.name}")
            result = run_strategy_backtest(strategy, candles, initial_capital=10000.0)
            print_backtest_report(result, strategy.name)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)

    else:
        # Strategy comparison (default)
        print("Comparing all strategies...")
        results = compare_strategies(candles, initial_capital=10000.0)
        print_strategy_comparison(results)

        # Also show signal-based for reference
        print("\n\nAlso testing signal-based approach...")
        config = {
            "rsi_enabled": True,
            "macd_enabled": True,
            "bollinger_enabled": True,
            "ema_crossover_enabled": True,
            "stochastic_enabled": False,
            "combine_method": "majority",
        }
        signal_result = run_backtest_on_training_data(
            signal_config=config,
            candle_period=CANDLE_PERIOD,
            initial_capital=10000.0
        )
        if signal_result:
            print(f"\nSignal-Based Result: {signal_result.total_return_pct:+.2f}% "
                  f"({signal_result.num_trades} trades, {signal_result.win_rate:.1f}% win rate)")
