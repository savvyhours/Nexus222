# Weight Calibration Skill

**Trigger:** Called by every NEXUS-II component before using any weight, threshold, or risk parameter.

## Purpose

Returns calibrated weights and thresholds tuned to the current market regime (TRENDING / MEAN_REVERTING / HIGH_VOL / LOW_VOL / CRISIS).

## Inputs (market state)

- `india_vix` — India VIX index value
- `nifty_change_pct` — Nifty 50 daily change %
- `fii_dii_flow` — Net FII/DII flow (₹ crore)
- `market_regime` — Detected regime (from RegimeDetector)
- `current_drawdown` — Current portfolio drawdown %
- `daily_pnl_pct` — Today's P&L as % of capital
- `agent_sharpe_30d` — Rolling 30-day Sharpe per agent
- `sector_momentum` — Per-sector momentum scores
- `market_breadth` — Advance/decline ratio
- `options_iv_percentile` — Average IV percentile across FnO universe
- `time_of_day` — IST time (HH:MM)
- `day_of_week` — Monday–Friday

## Outputs (JSON)

```json
{
  "market_regime": "TRENDING",
  "signal_weights": { "technical": 0.30, ... },
  "signal_threshold": 0.55,
  "agent_weights": { "scalper": 0.10, ... },
  "risk_thresholds": { "max_position_pct": 0.05, ... },
  "sl_tp_multipliers": { "intraday_sl_atr": 2.0, ... },
  "position_sizing": { "default_pct": 0.03, ... },
  "qlib_models": { "NSE500": {...}, ... },
  "kill_switch": false,
  "reasoning": "Market is trending with ADX > 25..."
}
```

## Cache TTL
- Market hours (09:15–15:30 IST): 15 minutes
- Outside market hours: 1 hour

## Safety Bounds (hard limits — LLM cannot override)
- `max_position_pct` ≤ 10%
- `max_sector_pct` ≤ 35%
- `max_daily_loss_pct` ≤ 3%
- `max_drawdown_pct` ≤ 12%
- `signal_threshold` ∈ [0.40, 0.80]

## Implementation
See [core/calibration/weight_calibration_agent.py](../../core/calibration/weight_calibration_agent.py)
