"""
NEXUS-II — Safety Bounds
Hard limits that the WeightCalibrationAgent LLM output CANNOT override.
These are the last line of defence before calibrated values are cached and used.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ── Absolute ceilings / floors ────────────────────────────────────────────────
# Risk thresholds
MAX_POSITION_PCT        = 0.10   # Never allocate > 10 % of capital to one stock
MAX_SECTOR_PCT          = 0.35   # Never exceed 35 % in a single sector
MAX_DAILY_LOSS_PCT      = 0.03   # Never allow > 3 % daily loss limit
MAX_DRAWDOWN_PCT        = 0.12   # Never allow > 12 % portfolio drawdown limit
MIN_MARGIN_BUFFER_PCT   = 0.10   # Always keep at least 10 % margin buffer

# Signal threshold
MIN_SIGNAL_THRESHOLD    = 0.40   # Never trade on a score below 0.40
MAX_SIGNAL_THRESHOLD    = 0.80   # Never demand a score above 0.80 (system would never trade)

# SL / TP ATR multipliers
MIN_SL_ATR_MULTIPLIER   = 0.50   # SL cannot be tighter than 0.5x ATR
MAX_SL_ATR_MULTIPLIER   = 5.00   # SL cannot be wider than 5x ATR
MIN_RR_RATIO            = 1.00   # Risk:Reward must be at least 1:1
MAX_RR_RATIO            = 10.00  # Cap at 1:10 (absurdly wide targets are a bug)

# Position sizing
MIN_POSITION_PCT        = 0.005  # Minimum meaningful position: 0.5 %
MAX_DEFAULT_POSITION    = 0.05   # Default position can never exceed 5 %

# Agent weights
MIN_AGENT_WEIGHT        = 0.0    # Zero is valid (muted agent)
MAX_SINGLE_AGENT_WEIGHT = 0.40   # No single agent can dominate > 40 % of vote


def enforce(cal: dict) -> dict:
    """
    Apply all hard safety bounds to a calibration dict returned by the LLM.
    Mutates and returns the same dict. Logs any values that were clamped.

    Args:
        cal: Raw calibration dict parsed from LLM JSON response.

    Returns:
        The same dict with out-of-bounds values clamped to safe limits.
    """
    cal = _enforce_risk_thresholds(cal)
    cal = _enforce_signal_threshold(cal)
    cal = _enforce_sl_tp(cal)
    cal = _enforce_position_sizing(cal)
    cal = _enforce_agent_weights(cal)
    cal = _enforce_signal_weights(cal)
    return cal


# ── Private helpers ────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float, label: str) -> float:
    clamped = max(lo, min(hi, value))
    if clamped != value:
        log.warning("SafetyBounds: %s clamped %.4f → %.4f", label, value, clamped)
    return clamped


def _enforce_risk_thresholds(cal: dict) -> dict:
    rt = cal.get("risk_thresholds", {})

    rt["max_position_pct"] = _clamp(
        rt.get("max_position_pct", 0.05), 0.001, MAX_POSITION_PCT, "max_position_pct"
    )
    rt["max_sector_pct"] = _clamp(
        rt.get("max_sector_pct", 0.25), 0.05, MAX_SECTOR_PCT, "max_sector_pct"
    )
    rt["max_daily_loss_pct"] = _clamp(
        rt.get("max_daily_loss_pct", 0.02), 0.005, MAX_DAILY_LOSS_PCT, "max_daily_loss_pct"
    )
    rt["max_drawdown_pct"] = _clamp(
        rt.get("max_drawdown_pct", 0.08), 0.02, MAX_DRAWDOWN_PCT, "max_drawdown_pct"
    )
    rt["margin_buffer_pct"] = max(
        rt.get("margin_buffer_pct", 0.20), MIN_MARGIN_BUFFER_PCT
    )

    # VIX thresholds must be logically ordered
    defensive = rt.get("vix_defensive_threshold", 22)
    halt = rt.get("vix_halt_threshold", 28)
    if halt <= defensive:
        log.warning(
            "SafetyBounds: vix_halt_threshold (%s) <= vix_defensive_threshold (%s). "
            "Resetting to defaults.",
            halt, defensive,
        )
        rt["vix_defensive_threshold"] = 22
        rt["vix_halt_threshold"] = 28

    cal["risk_thresholds"] = rt
    return cal


def _enforce_signal_threshold(cal: dict) -> dict:
    cal["signal_threshold"] = _clamp(
        cal.get("signal_threshold", 0.60),
        MIN_SIGNAL_THRESHOLD,
        MAX_SIGNAL_THRESHOLD,
        "signal_threshold",
    )
    return cal


def _enforce_sl_tp(cal: dict) -> dict:
    sl = cal.get("sl_tp_multipliers", {})

    for key in ("intraday_sl_atr", "positional_sl_atr", "trailing_stop_atr"):
        sl[key] = _clamp(
            sl.get(key, 2.0), MIN_SL_ATR_MULTIPLIER, MAX_SL_ATR_MULTIPLIER, key
        )

    sl["target_risk_reward"] = _clamp(
        sl.get("target_risk_reward", 2.0), MIN_RR_RATIO, MAX_RR_RATIO, "target_risk_reward"
    )

    # trailing stop must be <= intraday SL (otherwise it never fires before SL)
    if sl.get("trailing_stop_atr", 1.5) > sl.get("intraday_sl_atr", 2.0):
        log.warning(
            "SafetyBounds: trailing_stop_atr (%.2f) > intraday_sl_atr (%.2f). "
            "Clamping trailing stop to 75%% of intraday SL.",
            sl["trailing_stop_atr"], sl["intraday_sl_atr"],
        )
        sl["trailing_stop_atr"] = sl["intraday_sl_atr"] * 0.75

    cal["sl_tp_multipliers"] = sl
    return cal


def _enforce_position_sizing(cal: dict) -> dict:
    ps = cal.get("position_sizing", {})

    ps["default_pct"] = _clamp(
        ps.get("default_pct", 0.03), MIN_POSITION_PCT, MAX_DEFAULT_POSITION, "default_pct"
    )
    ps["max_pct"] = _clamp(
        ps.get("max_pct", 0.05), ps["default_pct"], MAX_POSITION_PCT, "position max_pct"
    )
    ps["high_conviction_pct"] = _clamp(
        ps.get("high_conviction_pct", 0.05),
        ps["default_pct"],
        MAX_POSITION_PCT,
        "high_conviction_pct",
    )
    ps["low_conviction_pct"] = _clamp(
        ps.get("low_conviction_pct", 0.01),
        MIN_POSITION_PCT,
        ps["default_pct"],
        "low_conviction_pct",
    )

    if "max_position_pct" in ps:
        ps["max_position_pct"] = _clamp(
            ps["max_position_pct"], 0.001, MAX_POSITION_PCT, "max_position_pct"
        )

    cal["position_sizing"] = ps
    return cal


def _enforce_agent_weights(cal: dict) -> dict:
    aw = cal.get("agent_weights", {})
    if not aw:
        return cal

    for agent, w in list(aw.items()):
        aw[agent] = _clamp(w, MIN_AGENT_WEIGHT, MAX_SINGLE_AGENT_WEIGHT, f"agent_weight[{agent}]")

    total = sum(aw.values())
    if total > 0:
        cal["agent_weights"] = {k: v / total for k, v in aw.items()}
    else:
        n = len(aw)
        log.error(
            "SafetyBounds: all agent_weights are 0. Resetting to equal weights (1/%d).", n
        )
        cal["agent_weights"] = {k: 1.0 / n for k in aw}

    return cal


class SafetyBounds:
    """
    Object-oriented wrapper around the module-level safety-bound functions.
    Used by tests and any caller that prefers an instance-based API.
    """

    def apply_signal_weights(self, raw: dict) -> dict:
        """Clamp negatives to 0, then normalise so weights sum to 1.0."""
        cal = {"signal_weights": dict(raw)}
        cal = _enforce_signal_weights(cal)
        return cal["signal_weights"]

    def apply_agent_weights(self, raw: dict) -> dict:
        """Clamp each weight to [0, MAX_SINGLE_AGENT_WEIGHT], then normalise."""
        cal = {"agent_weights": dict(raw)}
        cal = _enforce_agent_weights(cal)
        return cal["agent_weights"]

    def apply_risk_thresholds(self, raw: dict) -> dict:
        """
        Clamp signal_threshold to [MIN_SIGNAL_THRESHOLD, MAX_SIGNAL_THRESHOLD]
        and validate VIX ordering.
        """
        result = dict(raw)
        if "signal_threshold" in result:
            result["signal_threshold"] = _clamp(
                result["signal_threshold"],
                MIN_SIGNAL_THRESHOLD,
                MAX_SIGNAL_THRESHOLD,
                "signal_threshold",
            )
        return result

    def apply_position_sizing(self, raw: dict) -> dict:
        """Clamp position sizing fields to hard limits."""
        cal = {"position_sizing": dict(raw)}
        cal = _enforce_position_sizing(cal)
        return cal["position_sizing"]


def _enforce_signal_weights(cal: dict) -> dict:
    sw = cal.get("signal_weights", {})
    if not sw:
        return cal

    for key, w in list(sw.items()):
        if w < 0:
            log.warning(
                "SafetyBounds: signal_weight[%s] is negative (%.4f). Setting to 0.", key, w
            )
            sw[key] = 0.0

    total = sum(sw.values())
    if total > 0:
        cal["signal_weights"] = {k: v / total for k, v in sw.items()}
    else:
        n = len(sw)
        log.error("SafetyBounds: all signal_weights are 0. Resetting to equal weights.")
        cal["signal_weights"] = {k: 1.0 / n for k in sw}

    return cal
