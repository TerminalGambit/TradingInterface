# BTC Trading Interface

Real-time Bitcoin price tracking with technical indicators, paper trading simulation, and strategy backtesting.

## Quick Start

```bash
make install   # Install dependencies
make sim       # Start paper trading dashboard
```

## Commands

| Command | Description |
|---------|-------------|
| `make install` | Install Python dependencies |
| `make sim` | Start web-based paper trading dashboard |
| `make tracker` | CLI real-time price tracker with indicators |
| `make collect` | Run 24/7 data collector for backtesting |
| `make backtest` | Compare trading strategies on collected data |
| `make strategies` | Demo all trading strategies |

## Features

### Paper Trading Dashboard (`make sim`)
- Real-time BTC price from Binance API
- Candlestick chart with EMA, Bollinger Bands
- Buy/Sell simulation with portfolio tracking
- Trading signals from multiple indicators

### CLI Tracker (`make tracker`)
- Lightweight terminal-based price viewer
- Configurable indicators (EMA, BB, RSI, ATR, Support/Resistance)
- Low latency updates

### Trading Strategies (`make backtest`)
- **Mean Reversion**: Z-score based, trades oversold/overbought conditions
- **Momentum**: EMA crossover + Rate of Change
- **Breakout**: Support/resistance level breaks
- **Grid Trading**: Fixed price interval trading
- **Combined**: Multi-strategy weighted voting

### Technical Indicators
- EMA/SMA (Exponential/Simple Moving Average)
- Bollinger Bands
- RSI (Relative Strength Index)
- MACD
- ATR (Average True Range)
- Stochastic Oscillator
- Support/Resistance levels

## Requirements

- Python 3.10+
- Internet connection (Binance API)

## Project Structure

```
src/
├── app.py           # Dash web dashboard
├── btc_tracker.py   # CLI price tracker
├── indicators.py    # Technical indicators
├── signals.py       # Trading signal generators
├── strategies.py    # Trading strategies
├── backtester.py    # Backtesting engine
├── portfolio.py     # Portfolio management
└── data_collector.py # 24/7 data collection
```

## License

MIT
