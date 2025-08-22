# risk_manager.py
import json, os, datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

@dataclass
class SymbolConfig:
    symbol: str
    point_value: float
    max_position: int
    max_order_size: int
    min_stop_points: float
    max_stop_points: float

@dataclass
class RiskConfig:
    initial_equity: float
    risk_per_trade_pct: float
    daily_loss_limit_pct: float
    weekly_loss_limit_pct: float
    monthly_loss_limit_pct: float
    max_consecutive_losses: int
    cooldown_after_loss_minutes: int
    cooldown_after_daily_lock_minutes: int
    symbols: Dict[str, SymbolConfig]
    trading_day_cutoff: dt.time = dt.time(3, 0)  # 夜盘到凌晨3点算前一日

class RiskManager:
    def __init__(self, config: RiskConfig, state_path="./risk_state.json",
                 get_equity: Optional[Callable[[], float]] = None,
                 on_emergency_flatten: Optional[Callable[[], None]] = None,
                 on_disable_trading: Optional[Callable[[str], None]] = None):
        self.cfg = config
        self.state_path = state_path
        self.get_equity = get_equity or (lambda: self.cfg.initial_equity)
        self.on_emergency_flatten = on_emergency_flatten or (lambda: None)
        self.on_disable_trading = on_disable_trading or (lambda reason: None)
        self.state = {
            "daily_pnl": 0.0, "weekly_pnl": 0.0, "monthly_pnl": 0.0,
            "consecutive_losses": 0, "last_trade_time": None,
            "disabled_until": None,
        }
        self._load_state()

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f)

    def _load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path, "r") as f:
                self.state.update(json.load(f))

    def recommend_size(self, symbol: str, entry_price: float, stop_price: float):
        if symbol not in self.cfg.symbols:
            return False, 0, "symbol not configured"
        scfg = self.cfg.symbols[symbol]
        stop_points = abs(entry_price - stop_price)
        if not (scfg.min_stop_points <= stop_points <= scfg.max_stop_points):
            return False, 0, f"stop distance {stop_points} invalid"
        equity = self.get_equity()
        risk_amt = equity * self.cfg.risk_per_trade_pct
        lots = int(risk_amt / (stop_points * scfg.point_value))
        lots = min(lots, scfg.max_order_size, scfg.max_position)
        if lots <= 0:
            return False, 0, "risk too high for entry"
        if self.state.get("disabled_until"):
            until = dt.datetime.fromisoformat(self.state["disabled_until"])
            if dt.datetime.now() < until:
                return False, 0, "trading disabled until " + until.isoformat()
        return True, lots, "ok"

    def on_fill(self, symbol: str, side: str, price: float, qty: int, pnl: float = 0.0):
        self.state["daily_pnl"] += pnl
        self.state["weekly_pnl"] += pnl
        self.state["monthly_pnl"] += pnl
        if pnl < 0:
            self.state["consecutive_losses"] += 1
            cd = dt.datetime.now() + dt.timedelta(minutes=self.cfg.cooldown_after_loss_minutes)
            self.state["disabled_until"] = cd.isoformat()
        else:
            self.state["consecutive_losses"] = 0
        self._check_limits()
        self._save_state()

    def _check_limits(self):
        equity = self.get_equity()
        init = self.cfg.initial_equity
        if self.state["daily_pnl"] < -init * self.cfg.daily_loss_limit_pct:
            self._lock("daily loss limit hit")
        if self.state["weekly_pnl"] < -init * self.cfg.weekly_loss_limit_pct:
            self._lock("weekly loss limit hit")
        if self.state["monthly_pnl"] < -init * self.cfg.monthly_loss_limit_pct:
            self._lock("monthly loss limit hit")
        if self.state["consecutive_losses"] >= self.cfg.max_consecutive_losses:
            self._lock("max consecutive losses hit")

    def _lock(self, reason: str):
        cd = dt.datetime.now() + dt.timedelta(minutes=self.cfg.cooldown_after_daily_lock_minutes)
        self.state["disabled_until"] = cd.isoformat()
        self.on_emergency_flatten()
        self.on_disable_trading(reason)
        self._save_state()
