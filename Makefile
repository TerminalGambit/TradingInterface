.PHONY: install sim tracker collect backtest strategies clean help

# Default target
help:
	@echo "BTC Trading Interface"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install dependencies"
	@echo "  make sim         Start paper trading dashboard (web)"
	@echo "  make tracker     Start CLI price tracker with indicators"
	@echo "  make collect     Run 24/7 data collector"
	@echo "  make backtest    Run strategy backtester"
	@echo "  make strategies  Demo all trading strategies"
	@echo "  make clean       Remove cached files"
	@echo ""

# Install all dependencies
install:
	pip install requests numpy matplotlib dash plotly dash-bootstrap-components waitress

# Start the paper trading web dashboard
sim:
	cd src && python app.py

# Start CLI real-time tracker with indicators
tracker:
	cd src && python btc_tracker.py

# Run data collector for backtesting
collect:
	cd src && python data_collector.py

# Run backtester (compares all strategies)
backtest:
	cd src && python backtester.py

# Fetch historical data and backtest (e.g., make historical DAYS=30 INTERVAL=1h)
DAYS ?= 7
INTERVAL ?= 1h
historical:
	cd src && python fetch_historical.py $(DAYS) $(INTERVAL)

# Demo the trading strategies
strategies:
	cd src && python strategies.py

# Clean Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
