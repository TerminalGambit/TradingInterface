# BTC Trading System Roadmap

## Phase 4b: Simple Trading Strategies (Complete)

- [x] Mean Reversion strategy (z-score based)
- [x] Momentum strategy (EMA crossover + ROC)
- [x] Breakout strategy (support/resistance levels)
- [x] Grid Trading strategy (fixed price intervals)
- [x] Combined strategy (multi-strategy voting)
- [x] Strategy comparison framework in backtester

## Phase 5: ML Training (Complete)

- [x] Feature engineering from indicators (27 features)
- [x] Train model to predict price direction (Logistic, RF, GBM)
- [x] Compare ML vs rule-based signals
- [ ] Paper trade with ML predictions (optional future work)

### ML Results (30-day backtest)

| Strategy | Return | Trades | Win Rate |
|----------|--------|--------|----------|
| Momentum | +6.03% | 5 | 60% |
| Breakout | +3.76% | 11 | 64% |
| **ML Strategy** | **+3.14%** | **143** | **57%** |
| Mean Reversion | -10.56% | 15 | 53% |

Note: ML achieves 57% win rate but trades much more frequently.
Best accuracy ~51% (predicting price direction is hard!)

## Phase 6: Future Ideas

- [ ] Deep learning (LSTM, Transformer)
- [ ] Sentiment analysis from news/social
- [ ] Multi-asset portfolio optimization
- [ ] Live paper trading with ML signals
