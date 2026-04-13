"""
NEXUS-II v2.1 — Main Entry Point
AI-Powered Automated Trading System for Indian Markets (NSE/BSE)

10-Layer Architecture Startup Sequence
──────────────────────────────────────
Layer 0   WeightCalibrationAgent + RegimeDetector + MCPTools
Layer 0b  AsyncAnthropic client (shared by all LLM components)
Layer 1   Analysts: TechnicalAnalyst, SentimentAnalyst, FundamentalAnalyst, MacroAnalyst
Layer 2   Strategy Sub-Agents (10-agent Sharpe-weighted swarm)
Layer 3   Bull / Bear / Risk Researchers
Layer 4   Debate Arena
Layer 5   MasterOrchestrator + PortfolioManager
Layer 6   DynamicSignalScorer
Layer 7   Risk: PreTradeRiskChecker, PositionManager, DrawdownMonitor
Layer 8   TelegramBot
Layer 9   Execution: DhanExecutor (LIVE) or PaperTrader (PAPER)
Layer 10  GuardWatchdog + ReportGenerator
"""
from __future__ import annotations

import asyncio
import logging
import signal
import os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

import config.settings as cfg

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, cfg.LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("nexus-ii")

IST = ZoneInfo("Asia/Kolkata")

# ── Starting capital (from env or default ₹5,00,000) ─────────────────────
STARTING_CAPITAL = float(os.getenv("STARTING_CAPITAL", "500000"))

# ── Graceful shutdown ─────────────────────────────────────────────────────
_shutdown = asyncio.Event()


def _handle_signal(*_):
    log.warning("Shutdown signal received — stopping gracefully …")
    _shutdown.set()


# ── Market hours helpers ──────────────────────────────────────────────────

def _is_market_hours() -> bool:
    now = datetime.now(IST).time()
    return dtime(9, 15) <= now <= dtime(15, 30)


def _market_open_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")


# ── Component factory ─────────────────────────────────────────────────────

async def _build_components() -> dict:
    """
    Instantiate all 10 layers in dependency order.
    Returns a dict of named component instances.
    """
    log.info("⚙  Initialising components …")

    # ── Layer 0b: Shared Claude async client ─────────────────────────────
    # Must be created first — WeightCalibrationAgent depends on it.
    import anthropic
    claude_client = anthropic.AsyncAnthropic(api_key=cfg.CLAUDE_API_KEY)
    log.info("  ✓ Layer 0b AsyncAnthropic client (claude-sonnet-4-6)")

    # ── Layer 0c: MCP Tools (Dhan + Data + Compute) ──────────────────────
    # Must be created before WeightCalibrationAgent (market_data_tools dep).
    import dhanhq
    from core.mcp_tools.tool_registry import create_mcp_tools

    dhan_client = dhanhq.DhanHQ(
        client_id=cfg.DHAN_CLIENT_ID,
        access_token=cfg.DHAN_ACCESS_TOKEN,
    )
    mcp_tools = create_mcp_tools(
        dhan_client=dhan_client,
        starting_capital=STARTING_CAPITAL,
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_KEY", ""),
        nse_session_cookie=os.getenv("NSE_SESSION_COOKIE", ""),
        fred_api_key=os.getenv("FRED_API_KEY", ""),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", ""),
    )
    log.info("  ✓ Layer 0c MCPTools (DhanTools + DataTools + ComputeTools)")

    # ── Layer 0a: Weight Calibration + Regime Detector ────────────────────
    # Placed after 0b/0c because WeightCalibrationAgent requires both
    # claude_client and mcp_tools at construction time.
    from core.calibration.weight_calibration_agent import WeightCalibrationAgent
    from core.calibration.regime_detector import RegimeDetector

    calibration_agent = WeightCalibrationAgent(
        claude_client=claude_client,
        market_data_tools=mcp_tools,
    )
    regime_detector = RegimeDetector()
    log.info("  ✓ Layer 0a WeightCalibrationAgent + RegimeDetector")

    # ── Layer 1: Tier-1 Analysts ──────────────────────────────────────────
    from core.agents.analysts import (
        TechnicalAnalyst,
        SentimentAnalyst,
        FundamentalAnalyst,
        MacroAnalyst,
    )

    analysts = {
        "technical":   TechnicalAnalyst(claude_client, calibration_agent),
        "sentiment":   SentimentAnalyst(claude_client, calibration_agent),
        "fundamental": FundamentalAnalyst(claude_client, calibration_agent),
        "macro":       MacroAnalyst(claude_client, calibration_agent),
    }
    log.info("  ✓ Layer 1  4 Analysts (Technical, Sentiment, Fundamental, Macro)")

    # ── Layer 2: 10 Strategy Sub-Agents ──────────────────────────────────
    from core.agents import (
        ScalperAgent,
        TrendFollowerAgent,
        MeanReversionAgent,
        SentimentAgent,
        FundamentalsAgent,
        MacroAgent,
        OptionsAgent,
        PatternAgent,
        QuantAgent,
        ETFAgent,
    )

    strategy_agents = [
        ScalperAgent(calibration_agent, mcp_tools),
        TrendFollowerAgent(calibration_agent, mcp_tools),
        MeanReversionAgent(calibration_agent, mcp_tools),
        SentimentAgent(calibration_agent, mcp_tools),
        FundamentalsAgent(calibration_agent, mcp_tools),
        MacroAgent(calibration_agent, mcp_tools),
        OptionsAgent(calibration_agent, mcp_tools),
        PatternAgent(calibration_agent, mcp_tools),
        QuantAgent(calibration_agent, mcp_tools),
        ETFAgent(calibration_agent, mcp_tools),
    ]
    log.info("  ✓ Layer 2  10 Strategy Sub-Agents")

    # ── Layer 3: Tier-2 Researchers ───────────────────────────────────────
    from core.agents.researchers import BullResearcher, BearResearcher, RiskResearcher

    researchers = {
        "bull": BullResearcher(calibration_agent, claude_client),
        "bear": BearResearcher(calibration_agent, claude_client),
        "risk": RiskResearcher(calibration_agent, claude_client),
    }
    log.info("  ✓ Layer 3  Bull / Bear / Risk Researchers")

    # ── Layer 4: Debate Arena ─────────────────────────────────────────────
    from core.agents.debate import DebateArena

    debate_arena = DebateArena(calibration_agent, claude_client)
    log.info("  ✓ Layer 4  Debate Arena")

    # ── Layer 5: Orchestrator + Portfolio Manager ─────────────────────────
    from core.agents.master_orchestrator import MasterOrchestrator
    from core.agents.portfolio_manager import PortfolioManager

    orchestrator = MasterOrchestrator(
        agents=strategy_agents,
        calibration_agent=calibration_agent,
        claude_client=claude_client,
    )
    portfolio_manager = PortfolioManager(
        calibration_agent=calibration_agent,
        claude_client=claude_client,
    )
    log.info("  ✓ Layer 5  MasterOrchestrator + PortfolioManager")

    # ── Layer 6: Signal Engine ────────────────────────────────────────────
    from core.signal_engine import DynamicSignalScorer

    signal_scorer = DynamicSignalScorer(calibration_agent)
    log.info("  ✓ Layer 6  DynamicSignalScorer")

    # ── Layer 7: Risk Management ──────────────────────────────────────────
    from core.risk import PreTradeRiskChecker, PositionManager, DrawdownMonitor

    pre_trade_checks = PreTradeRiskChecker(
        calibration_agent=calibration_agent,
        mcp_tools=mcp_tools,
        total_capital=STARTING_CAPITAL,
    )
    position_manager = PositionManager(calibration_agent=calibration_agent)
    drawdown_monitor = DrawdownMonitor(
        calibration_agent=calibration_agent,
        starting_capital=STARTING_CAPITAL,
    )
    log.info("  ✓ Layer 7  Risk (PreTradeRiskChecker + PositionManager + DrawdownMonitor)")

    # ── Layer 8: Telegram Approval Bot ───────────────────────────────────
    from core.monitoring.telegram_bot import TelegramBot

    telegram_bot = TelegramBot(
        bot_token=cfg.TELEGRAM_BOT_TOKEN,
        chat_id=cfg.TELEGRAM_CHAT_ID,
        trading_mode=cfg.TRADING_MODE,
    )
    log.info("  ✓ Layer 8  TelegramBot")

    # ── Layer 9: Execution Engine ─────────────────────────────────────────
    if cfg.TRADING_MODE == "LIVE":
        from core.execution.dhan_executor import DhanExecutor
        executor = DhanExecutor(dhan_tools=mcp_tools.dhan)
        log.info("  ✓ Layer 9  DhanExecutor (LIVE trading)")
    else:
        from core.execution.paper_trader import PaperTrader
        executor = PaperTrader()
        log.info("  ✓ Layer 9  PaperTrader (paper trading)")

    # ── Layer 10: Monitoring ──────────────────────────────────────────────
    from core.monitoring.guard_watchdog import GuardWatchdog
    from core.monitoring.report_generator import ReportGenerator

    # Connect drawdown monitor kill-switch callback to watchdog alert
    from supabase import create_client as _sb_create
    supabase_client = _sb_create(cfg.SUPABASE_URL, cfg.SUPABASE_SERVICE_ROLE_KEY)

    watchdog = GuardWatchdog(
        mcp_tools=mcp_tools,
        drawdown_monitor=drawdown_monitor,
        telegram_bot=telegram_bot,
    )
    report_generator = ReportGenerator(
        supabase_client=supabase_client,
        telegram_bot=telegram_bot,
    )
    log.info("  ✓ Layer 10 GuardWatchdog + ReportGenerator")

    log.info("⚙  All layers initialised. System ready.")

    return {
        "calibration_agent": calibration_agent,
        "regime_detector": regime_detector,
        "mcp_tools": mcp_tools,
        "claude_client": claude_client,
        "analysts": analysts,
        "strategy_agents": strategy_agents,
        "researchers": researchers,
        "debate_arena": debate_arena,
        "orchestrator": orchestrator,
        "portfolio_manager": portfolio_manager,
        "signal_scorer": signal_scorer,
        "pre_trade_checks": pre_trade_checks,
        "position_manager": position_manager,
        "drawdown_monitor": drawdown_monitor,
        "telegram_bot": telegram_bot,
        "executor": executor,
        "watchdog": watchdog,
        "report_generator": report_generator,
        "supabase_client": supabase_client,
    }


# ── Single trading cycle ──────────────────────────────────────────────────

async def _run_cycle(components: dict) -> None:
    """
    One full trading cycle:
      1. Detect regime & warm calibration cache
      2. Run orchestrator (all 10 agents concurrently per symbol)
      3. For consensus signals → analysts → research → debate → PM review
      4. Pre-trade risk checks
      5. Telegram approval (LIVE) or auto-execute (PAPER)
      6. Send to executor
    """
    from config.universe import MARKET_UNIVERSES

    cal          = components["calibration_agent"]
    mcp          = components["mcp_tools"]
    orchestrator = components["orchestrator"]
    debate_arena = components["debate_arena"]
    researchers  = components["researchers"]
    pm           = components["portfolio_manager"]
    pre_trade    = components["pre_trade_checks"]
    telegram_bot = components["telegram_bot"]
    executor     = components["executor"]
    analysts     = components["analysts"]

    log.info("── Trading cycle start ──────────────────────────────")

    # 1. Calibrate weights for current regime (cache-aware; LLM only every 15 min)
    await cal.calibrate()

    # 2. Determine universe for this cycle
    universe_symbols: list[str] = MARKET_UNIVERSES.get("NIFTY50", [])

    # 3. Fetch market + sentiment data for all symbols concurrently
    market_data_tasks = {sym: mcp.dhan.get_market_snapshot(sym) for sym in universe_symbols}
    sentiment_tasks   = {sym: mcp.data.get_news_headlines(sym) for sym in universe_symbols}

    market_data_map = {}
    sentiment_data_map = {}
    for sym in universe_symbols:
        try:
            market_data_map[sym]    = await market_data_tasks[sym]
            sentiment_data_map[sym] = {"headlines": await sentiment_tasks[sym], "score": 0.0}
        except Exception as exc:
            log.warning("Data fetch failed for %s: %s", sym, exc)

    if not market_data_map:
        log.warning("No market data fetched — skipping cycle")
        return

    # 4. Orchestrator runs all 10 agents concurrently → ConsensusSignal list
    consensus_signals = await orchestrator.run_cycle(
        universe=list(market_data_map.keys()),
        market_data_map=market_data_map,
        sentiment_data_map=sentiment_data_map,
    )

    if not consensus_signals:
        log.info("No consensus signals this cycle.")
        return

    log.info("%d consensus signal(s) → research + debate …", len(consensus_signals))

    for cs in consensus_signals:
        sym = cs.symbol
        md = market_data_map.get(sym, {})

        try:
            # 5. Analyst reports (concurrent across 4 analysts)
            analyst_results = await asyncio.gather(
                *[a.analyze(sym, md) for a in analysts.values()],
                return_exceptions=True,
            )
            analyst_reports = [
                r for r in analyst_results if not isinstance(r, Exception)
            ]
            if not analyst_reports:
                log.warning("%s: all analyst calls failed — skipping", sym)
                continue

            # 6. Research (concurrent bull + bear + risk)
            bull_thesis, bear_thesis, risk_assessment = await asyncio.gather(
                researchers["bull"].research(sym, analyst_reports),
                researchers["bear"].research(sym, analyst_reports),
                researchers["risk"].assess(sym, analyst_reports),
            )

            # 7. Debate Arena (3-round structured debate)
            debate_verdict = await debate_arena.run(
                symbol=sym,
                bull=bull_thesis,
                bear=bear_thesis,
                risk=risk_assessment,
                analyst_reports=analyst_reports,
            )

            # 8. Portfolio Manager final review
            trade_proposal = await pm.review(cs, debate_verdict)
            if trade_proposal is None:
                log.info("  %s: PM rejected", sym)
                continue

            # 9. Pre-trade risk checks
            check_result = await pre_trade.check_all(trade_proposal)
            if not check_result.approved:
                log.info("  %s: Pre-trade check failed — %s", sym, check_result.reason)
                continue

            # 10. Telegram approval for LIVE; auto-execute for PAPER
            if cfg.TRADING_MODE == "LIVE":
                confirmed = await telegram_bot.request_approval(trade_proposal)
                if not confirmed:
                    log.info("  %s: Telegram rejected by trader", sym)
                    continue

            # 11. Execute order
            order_result = await executor.execute(trade_proposal)
            log.info("  %s: Order placed → %s", sym, order_result)

        except Exception as exc:
            log.exception("  %s: Cycle error — %s", sym, exc)
            continue

    log.info("── Trading cycle complete ───────────────────────────")


# ── Main loop ─────────────────────────────────────────────────────────────

async def main() -> None:
    log.info("🚀 NEXUS-II v2.1 starting | mode=%s | %s", cfg.TRADING_MODE, _market_open_str())

    # Register OS signals for graceful shutdown
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows does not support SIGTERM via add_signal_handler
            pass

    components = await _build_components()

    # Start watchdog as background task
    watchdog_task = asyncio.create_task(
        components["watchdog"].start(),
        name="watchdog",
    )

    log.info("✅ System ready. Entering main loop …")
    cycle_interval_seconds = 60  # run every 60s during market hours

    try:
        while not _shutdown.is_set():
            if _is_market_hours():
                try:
                    await _run_cycle(components)
                except Exception as exc:
                    log.exception("Cycle error (continuing): %s", exc)
            else:
                log.debug("Market closed — sleeping …")

            try:
                await asyncio.wait_for(_shutdown.wait(), timeout=cycle_interval_seconds)
            except asyncio.TimeoutError:
                pass  # normal — loop again

    finally:
        log.info("Shutting down …")
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass
        log.info("NEXUS-II stopped. Goodbye.")


if __name__ == "__main__":
    asyncio.run(main())
