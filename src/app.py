"""
BTC Paper Trading Emulator - Dash App
Run with: python app.py
"""

from dash import Dash, html, dcc, callback, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import time
from datetime import datetime
from collections import deque

from portfolio import Portfolio, PortfolioManager
from indicators import (
    calculate_ema, calculate_bollinger_bands, calculate_rsi, calculate_atr
)
from signals import SignalGenerator
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"
SYMBOL = "BTCUSDT"
MAX_HISTORY = 200  # Keep last 200 ticks
CANDLE_PERIOD = 15  # seconds per candle


# =============================================================================
# GLOBAL STATE
# =============================================================================
price_history = deque(maxlen=MAX_HISTORY)
portfolio_manager = PortfolioManager()
portfolio = portfolio_manager.load()
signal_generator = SignalGenerator({
    "rsi_enabled": True,
    "macd_enabled": True,
    "bollinger_enabled": True,
    "ema_crossover_enabled": True,
    "stochastic_enabled": False,
    "combine_method": "majority",
})


def fetch_btc_price() -> float:
    """Fetch current BTC price from Binance."""
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": SYMBOL}, timeout=5)
        response.raise_for_status()
        return float(response.json()["price"])
    except Exception as e:
        print(f"Error fetching price: {e}")
        return None


def aggregate_to_candles(ticks: list) -> list:
    """Aggregate ticks into OHLC candles."""
    if not ticks:
        return []

    candles = []
    current_candle = None
    candle_start = None

    for tick in ticks:
        epoch = tick["epoch"]
        price = tick["price"]

        if candle_start is None:
            candle_start = epoch
            current_candle = {
                "time": datetime.fromtimestamp(epoch),
                "open": price, "high": price, "low": price, "close": price
            }
        elif epoch - candle_start >= CANDLE_PERIOD:
            candles.append(current_candle)
            candle_start = epoch
            current_candle = {
                "time": datetime.fromtimestamp(epoch),
                "open": price, "high": price, "low": price, "close": price
            }
        else:
            current_candle["high"] = max(current_candle["high"], price)
            current_candle["low"] = min(current_candle["low"], price)
            current_candle["close"] = price

    if current_candle:
        candles.append(current_candle)

    return candles


# =============================================================================
# DASH APP
# =============================================================================
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("📈 BTC Paper Trading", className="text-center my-3"),
        ])
    ]),

    # Price display
    dbc.Row([
        dbc.Col([
            html.Div(id="price-display", className="text-center mb-3",
                     style={"fontSize": "2.5rem", "fontWeight": "bold"})
        ])
    ]),

    # Main content
    dbc.Row([
        # Left: Chart
        dbc.Col([
            dcc.Graph(id="price-chart", style={"height": "500px"}),
        ], width=8),

        # Right: Portfolio & Trading
        dbc.Col([
            # Portfolio Card
            dbc.Card([
                dbc.CardHeader("💰 Portfolio"),
                dbc.CardBody([
                    html.Div(id="portfolio-display"),
                ])
            ], className="mb-3"),

            # Trading Card
            dbc.Card([
                dbc.CardHeader("🔄 Trade"),
                dbc.CardBody([
                    dbc.RadioItems(
                        id="trade-action",
                        options=[
                            {"label": "Buy", "value": "buy"},
                            {"label": "Sell", "value": "sell"}
                        ],
                        value="buy",
                        inline=True,
                        className="mb-3"
                    ),
                    dbc.Input(
                        id="trade-amount",
                        type="number",
                        placeholder="USD amount (for buy) / BTC amount (for sell)",
                        className="mb-3"
                    ),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Execute", id="btn-trade", color="primary",
                                       className="w-100")
                        ]),
                        dbc.Col([
                            dbc.Button("Max", id="btn-max", color="secondary",
                                       className="w-100")
                        ]),
                    ], className="mb-2"),
                    dbc.Button("🔄 Reset Portfolio", id="btn-reset", color="danger",
                               outline=True, className="w-100 mt-2"),
                    html.Div(id="trade-message", className="mt-2")
                ])
            ], className="mb-3"),

            # Indicators Card
            dbc.Card([
                dbc.CardHeader("📊 Indicators"),
                dbc.CardBody([
                    html.Div(id="indicators-display")
                ])
            ], className="mb-3"),

            # Signal Card
            dbc.Card([
                dbc.CardHeader("🎯 Trading Signal"),
                dbc.CardBody([
                    html.Div(id="signal-display")
                ])
            ]),
        ], width=4),
    ]),

    # Trade History
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Trade History"),
                dbc.CardBody([
                    html.Div(id="trade-history")
                ])
            ])
        ])
    ], className="mt-3"),

    # Auto-refresh interval
    dcc.Interval(id="interval", interval=2000, n_intervals=0),

    # Store for current price
    dcc.Store(id="current-price-store"),

], fluid=True)


# =============================================================================
# CALLBACKS
# =============================================================================

@callback(
    Output("current-price-store", "data"),
    Output("price-display", "children"),
    Output("price-chart", "figure"),
    Output("portfolio-display", "children"),
    Output("indicators-display", "children"),
    Output("signal-display", "children"),
    Input("interval", "n_intervals"),
)
def update_data(n):
    global price_history, portfolio

    # Fetch new price
    price = fetch_btc_price()
    if price:
        price_history.append({
            "epoch": time.time(),
            "price": price,
            "timestamp": datetime.now()
        })

    current_price = price if price else (price_history[-1]["price"] if price_history else 0)

    # Price display
    if len(price_history) > 1:
        change = ((current_price - price_history[0]["price"]) / price_history[0]["price"]) * 100
        color = "lime" if change >= 0 else "red"
        price_display = html.Span([
            f"${current_price:,.2f} ",
            html.Span(f"({change:+.2f}%)", style={"color": color, "fontSize": "1.5rem"})
        ])
    else:
        price_display = f"${current_price:,.2f}"

    # Build chart
    fig = build_chart(list(price_history), current_price)

    # Portfolio display
    portfolio = portfolio_manager.load()  # Reload in case of changes
    total_value = portfolio.total_value(current_price)
    unrealized = portfolio.unrealized_pnl(current_price)
    btc_value = portfolio.btc_balance * current_price

    pnl_color = "lime" if unrealized >= 0 else "red"
    portfolio_display = html.Div([
        html.P([html.Strong("Total Value: "), f"${total_value:,.2f}"]),
        html.P([html.Strong("USD: "), f"${portfolio.usd_balance:,.2f}"]),
        html.P([html.Strong("BTC: "), f"{portfolio.btc_balance:.6f}",
                html.Span(f" (${btc_value:,.2f})", style={"color": "gray"})]),
        html.P([html.Strong("Unrealized P&L: "),
                html.Span(f"${unrealized:,.2f}", style={"color": pnl_color})]),
    ])

    # Indicators display
    indicators_display = build_indicators(list(price_history))

    # Signal display
    signal_display = build_signal_display(list(price_history))

    return current_price, price_display, fig, portfolio_display, indicators_display, signal_display


def build_chart(ticks, current_price):
    """Build candlestick chart with indicators."""
    candles = aggregate_to_candles(ticks)

    fig = make_subplots(rows=1, cols=1)

    if len(candles) >= 2:
        times = [c["time"] for c in candles]
        opens = [c["open"] for c in candles]
        highs = [c["high"] for c in candles]
        lows = [c["low"] for c in candles]
        closes = [c["close"] for c in candles]

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=times,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            name="BTC",
            increasing_line_color="lime",
            decreasing_line_color="red"
        ))

        closes_arr = np.array(closes)

        # EMA
        if len(closes) >= 10:
            ema = calculate_ema(closes_arr, 10)
            fig.add_trace(go.Scatter(
                x=times, y=ema,
                mode="lines",
                name="EMA(10)",
                line=dict(color="orange", width=1.5)
            ))

        # Bollinger Bands
        if len(closes) >= 20:
            bands = calculate_bollinger_bands(closes_arr, window=20, num_std=2)
            pad = len(closes) - len(bands["middle"])
            band_times = times[pad:]

            fig.add_trace(go.Scatter(
                x=band_times, y=bands["upper"],
                mode="lines",
                name="BB Upper",
                line=dict(color="rgba(100,100,255,0.5)", width=1)
            ))
            fig.add_trace(go.Scatter(
                x=band_times, y=bands["lower"],
                mode="lines",
                name="BB Lower",
                line=dict(color="rgba(100,100,255,0.5)", width=1),
                fill="tonexty",
                fillcolor="rgba(100,100,255,0.1)"
            ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=30, b=50),
        xaxis_title="Time",
        yaxis_title="Price (USD)"
    )

    return fig


def build_signal_display(ticks):
    """Build trading signal display."""
    if len(ticks) < 30:
        return html.P(f"Need more data ({len(ticks)}/30 ticks)...")

    prices = np.array([t["price"] for t in ticks])

    # Get current signal
    signal_result = signal_generator.get_current_signal(prices, prices, prices)

    action = signal_result["action"]
    details = signal_result["details"]

    # Style based on action
    if action == "BUY":
        color = "lime"
        icon = "🟢"
    elif action == "SELL":
        color = "red"
        icon = "🔴"
    else:
        color = "gray"
        icon = "⚪"

    # Build details
    detail_items = []
    for name, sig in details.items():
        sig_icon = "🟢" if sig == 1 else "🔴" if sig == -1 else "⚪"
        sig_text = "BUY" if sig == 1 else "SELL" if sig == -1 else "HOLD"
        detail_items.append(html.P(f"{sig_icon} {name}: {sig_text}", style={"margin": "2px 0", "fontSize": "0.85rem"}))

    return html.Div([
        html.H3(f"{icon} {action}", style={"color": color, "textAlign": "center", "margin": "10px 0"}),
        html.Hr(),
        html.P(html.Strong("Individual Signals:"), style={"marginBottom": "5px"}),
        *detail_items
    ])


def build_indicators(ticks):
    """Build indicators display."""
    if len(ticks) < 15:
        return html.P(f"Need more data ({len(ticks)}/15 ticks)...")

    prices = np.array([t["price"] for t in ticks])

    indicators = []

    # RSI
    rsi = calculate_rsi(prices, period=14)
    if len(rsi) > 0:
        current_rsi = rsi[-1]
        rsi_status = "🔴 OB" if current_rsi > 70 else "🟢 OS" if current_rsi < 30 else "⚪"
        indicators.append(html.P([html.Strong("RSI: "), f"{current_rsi:.1f} {rsi_status}"]))

    # ATR
    atr = calculate_atr(prices, prices, prices, period=14)
    if len(atr) > 0:
        current_atr = atr[-1]
        atr_pct = (current_atr / prices[-1]) * 100
        indicators.append(html.P([html.Strong("ATR: "), f"${current_atr:.2f} ({atr_pct:.2f}%)"]))

    # Session change
    change = ((prices[-1] - prices[0]) / prices[0]) * 100
    change_color = "lime" if change >= 0 else "red"
    indicators.append(html.P([
        html.Strong("Session: "),
        html.Span(f"{change:+.2f}%", style={"color": change_color})
    ]))

    return html.Div(indicators)


@callback(
    Output("trade-message", "children"),
    Output("trade-history", "children"),
    Input("btn-trade", "n_clicks"),
    Input("btn-max", "n_clicks"),
    Input("btn-reset", "n_clicks"),
    State("trade-action", "value"),
    State("trade-amount", "value"),
    State("current-price-store", "data"),
    prevent_initial_call=True
)
def handle_trade(trade_clicks, max_clicks, reset_clicks, action, amount, current_price):
    global portfolio

    triggered = ctx.triggered_id
    message = ""

    portfolio = portfolio_manager.load()

    if triggered == "btn-reset":
        portfolio = portfolio_manager.reset()
        message = dbc.Alert("Portfolio reset to $10,000", color="info")

    elif triggered == "btn-max":
        if current_price:
            if action == "buy":
                trade = portfolio.buy_max(current_price)
                if trade:
                    portfolio_manager.save(portfolio)
                    message = dbc.Alert(f"Bought {trade.amount:.6f} BTC", color="success")
                else:
                    message = dbc.Alert("No USD available", color="warning")
            else:
                trade = portfolio.sell_all(current_price)
                if trade:
                    portfolio_manager.save(portfolio)
                    message = dbc.Alert(f"Sold {trade.amount:.6f} BTC", color="success")
                else:
                    message = dbc.Alert("No BTC to sell", color="warning")

    elif triggered == "btn-trade":
        if current_price and amount:
            if action == "buy":
                trade = portfolio.buy(current_price, usd_amount=float(amount))
                if trade:
                    portfolio_manager.save(portfolio)
                    message = dbc.Alert(f"Bought {trade.amount:.6f} BTC for ${amount}", color="success")
                else:
                    message = dbc.Alert("Insufficient USD", color="danger")
            else:
                trade = portfolio.sell(current_price, btc_amount=float(amount))
                if trade:
                    portfolio_manager.save(portfolio)
                    message = dbc.Alert(f"Sold {amount} BTC for ${trade.value:,.2f}", color="success")
                else:
                    message = dbc.Alert("Insufficient BTC", color="danger")
        else:
            message = dbc.Alert("Enter an amount", color="warning")

    # Build trade history table
    portfolio = portfolio_manager.load()
    if portfolio.trades:
        rows = []
        for t in reversed(portfolio.trades[-10:]):  # Last 10 trades
            color = "lime" if t["action"] == "BUY" else "red"
            rows.append(html.Tr([
                html.Td(t["timestamp"][11:19]),
                html.Td(t["action"], style={"color": color}),
                html.Td(f"${t['price']:,.2f}"),
                html.Td(f"{t['amount']:.6f}"),
                html.Td(f"${t['value']:,.2f}"),
            ]))

        history = dbc.Table([
            html.Thead(html.Tr([
                html.Th("Time"), html.Th("Action"), html.Th("Price"),
                html.Th("BTC"), html.Th("Value")
            ])),
            html.Tbody(rows)
        ], bordered=True, hover=True, size="sm", color="dark")
    else:
        history = html.P("No trades yet", className="text-muted")

    return message, history


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    print("Starting BTC Paper Trading app...")
    print("Open http://127.0.0.1:8050 in your browser")

    # Use waitress instead of Flask's built-in server to avoid werkzeug issues
    try:
        from waitress import serve
        serve(app.server, host="127.0.0.1", port=8050)
    except ImportError:
        print("Install waitress for better compatibility: pip install waitress")
        print("Falling back to Flask server...")
        app.run(debug=False, host="127.0.0.1", port=8050, use_reloader=False)
