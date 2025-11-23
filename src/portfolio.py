"""
Portfolio management module for paper trading.
Handles positions, balance, P&L, and trade history.
"""

import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Trade:
    timestamp: str
    action: str  # "BUY" or "SELL"
    price: float
    amount: float  # BTC amount
    value: float  # USD value
    balance_after: float
    btc_after: float


@dataclass
class Portfolio:
    usd_balance: float = 10000.0  # Starting balance
    btc_balance: float = 0.0
    trades: list = None
    created_at: str = None

    def __post_init__(self):
        if self.trades is None:
            self.trades = []
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

    def buy(self, price: float, usd_amount: Optional[float] = None, btc_amount: Optional[float] = None) -> Optional[Trade]:
        """
        Buy BTC. Specify either usd_amount or btc_amount.
        Returns Trade if successful, None if insufficient funds.
        """
        if usd_amount is not None:
            if usd_amount > self.usd_balance:
                return None
            btc_to_buy = usd_amount / price
            cost = usd_amount
        elif btc_amount is not None:
            cost = btc_amount * price
            if cost > self.usd_balance:
                return None
            btc_to_buy = btc_amount
        else:
            return None

        self.usd_balance -= cost
        self.btc_balance += btc_to_buy

        trade = Trade(
            timestamp=datetime.now().isoformat(),
            action="BUY",
            price=price,
            amount=btc_to_buy,
            value=cost,
            balance_after=self.usd_balance,
            btc_after=self.btc_balance
        )
        self.trades.append(asdict(trade))
        return trade

    def sell(self, price: float, btc_amount: Optional[float] = None, usd_amount: Optional[float] = None) -> Optional[Trade]:
        """
        Sell BTC. Specify either btc_amount or usd_amount (target USD to receive).
        Returns Trade if successful, None if insufficient BTC.
        """
        if btc_amount is not None:
            if btc_amount > self.btc_balance:
                return None
            btc_to_sell = btc_amount
            revenue = btc_to_sell * price
        elif usd_amount is not None:
            btc_to_sell = usd_amount / price
            if btc_to_sell > self.btc_balance:
                return None
            revenue = usd_amount
        else:
            return None

        self.btc_balance -= btc_to_sell
        self.usd_balance += revenue

        trade = Trade(
            timestamp=datetime.now().isoformat(),
            action="SELL",
            price=price,
            amount=btc_to_sell,
            value=revenue,
            balance_after=self.usd_balance,
            btc_after=self.btc_balance
        )
        self.trades.append(asdict(trade))
        return trade

    def sell_all(self, price: float) -> Optional[Trade]:
        """Sell all BTC holdings."""
        if self.btc_balance <= 0:
            return None
        return self.sell(price, btc_amount=self.btc_balance)

    def buy_max(self, price: float) -> Optional[Trade]:
        """Buy as much BTC as possible with current USD balance."""
        if self.usd_balance <= 0:
            return None
        return self.buy(price, usd_amount=self.usd_balance)

    def total_value(self, current_price: float) -> float:
        """Calculate total portfolio value in USD."""
        return self.usd_balance + (self.btc_balance * current_price)

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L from current BTC position."""
        if self.btc_balance <= 0 or not self.trades:
            return 0.0

        # Find average buy price for current position
        total_btc_bought = 0.0
        total_cost = 0.0

        for trade in self.trades:
            if trade["action"] == "BUY":
                total_btc_bought += trade["amount"]
                total_cost += trade["value"]
            else:  # SELL
                # Reduce proportionally
                if total_btc_bought > 0:
                    ratio = trade["amount"] / total_btc_bought
                    total_cost *= (1 - ratio)
                    total_btc_bought -= trade["amount"]

        if total_btc_bought <= 0:
            return 0.0

        avg_buy_price = total_cost / total_btc_bought
        return self.btc_balance * (current_price - avg_buy_price)

    def realized_pnl(self) -> float:
        """Calculate realized P&L from closed trades."""
        # Simple calculation: current USD - initial USD - value of remaining BTC at cost
        initial_balance = 10000.0  # Hardcoded initial
        return self.usd_balance - initial_balance + self._btc_cost_basis()

    def _btc_cost_basis(self) -> float:
        """Calculate cost basis of current BTC holdings."""
        if self.btc_balance <= 0:
            return 0.0

        total_btc = 0.0
        total_cost = 0.0

        for trade in self.trades:
            if trade["action"] == "BUY":
                total_btc += trade["amount"]
                total_cost += trade["value"]
            else:
                if total_btc > 0:
                    ratio = trade["amount"] / total_btc
                    total_cost *= (1 - ratio)
                    total_btc -= trade["amount"]

        return total_cost

    def to_dict(self) -> dict:
        """Convert portfolio to dictionary for JSON serialization."""
        return {
            "usd_balance": self.usd_balance,
            "btc_balance": self.btc_balance,
            "trades": self.trades,
            "created_at": self.created_at
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Portfolio":
        """Create Portfolio from dictionary."""
        return cls(
            usd_balance=data["usd_balance"],
            btc_balance=data["btc_balance"],
            trades=data["trades"],
            created_at=data["created_at"]
        )


class PortfolioManager:
    """Handles persistence of portfolio data."""

    def __init__(self, filepath: Path = None):
        if filepath is None:
            filepath = Path(__file__).parent / "portfolio_state.json"
        self.filepath = filepath

    def load(self) -> Portfolio:
        """Load portfolio from file or create new one."""
        if self.filepath.exists():
            with open(self.filepath, "r") as f:
                data = json.load(f)
                return Portfolio.from_dict(data)
        return Portfolio()

    def save(self, portfolio: Portfolio):
        """Save portfolio to file."""
        with open(self.filepath, "w") as f:
            json.dump(portfolio.to_dict(), f, indent=2)

    def reset(self, initial_balance: float = 10000.0) -> Portfolio:
        """Reset portfolio to initial state."""
        portfolio = Portfolio(usd_balance=initial_balance)
        self.save(portfolio)
        return portfolio
