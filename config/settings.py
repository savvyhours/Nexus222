"""
NEXUS-II — System Configuration
All environment-specific settings loaded from .env
"""
import os
from zoneinfo import ZoneInfo

# ── Trading Mode ──────────────────────────────────────────────────────────
TRADING_MODE = os.getenv("TRADING_MODE", "PAPER_TRADE")  # PAPER_TRADE | LIVE
IST = ZoneInfo("Asia/Kolkata")

# ── API Keys ──────────────────────────────────────────────────────────────
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
DHAN_ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
DHAN_CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# ── External Data API Keys ────────────────────────────────────────────────
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "")   # FOREX + commodities
FRED_API_KEY      = os.getenv("FRED_API_KEY", "")         # Federal Reserve macro data
FINNHUB_API_KEY   = os.getenv("FINNHUB_API_KEY", "")      # Real-time news + earnings

# ── LLM Models ────────────────────────────────────────────────────────────
LLM_MAIN = "claude-sonnet-4-6"       # All tiers: analysts, debate, PM, calibration
LLM_FAST = "claude-haiku-4-5-20251001"  # High-throughput NLP (entity extraction)

# ── Market Schedule (IST) ─────────────────────────────────────────────────
MARKET_OPEN = "09:15"
MARKET_CLOSE = "15:30"
INTRADAY_SQUAREOFF = "15:10"
PRE_MARKET_START = "08:00"

# ── Weight Calibration Cache ──────────────────────────────────────────────
CALIBRATION_TTL_MARKET_HOURS = 900   # 15 min during market hours
CALIBRATION_TTL_OFF_HOURS = 3600     # 1 hour outside market hours

# ── Options Chain Rate Limit ──────────────────────────────────────────────
OPTIONS_CHAIN_RATE_LIMIT_SECONDS = 150  # 1 request per 150 seconds

# ── Logging ───────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
