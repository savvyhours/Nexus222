"""
NEXUS-II — Trading Universe Definition
Defines stock universes per market segment.
"""

# Top 20 most liquid NSE stocks (for ScalperAgent)
NSE_TOP20_LIQUID = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
    "LT", "WIPRO", "HCLTECH", "AXISBANK", "ASIANPAINT",
    "MARUTI", "BAJFINANCE", "TITAN", "ULTRACEMCO", "NTPC",
]

# Nifty 50 constituents (for NIFTY50 Qlib model)
NIFTY50 = [
    "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BPCL", "BHARTIARTL",
    "BRITANNIA", "CIPLA", "COALINDIA", "DIVISLAB", "DRREDDY",
    "EICHERMOT", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE",
    "HEROMOTOCO", "HINDALCO", "HINDUNILVR", "ICICIBANK", "ITC",
    "INDUSINDBK", "INFY", "JSWSTEEL", "KOTAKBANK", "LT",
    "LTIM", "M&M", "MARUTI", "NESTLEIND", "NTPC",
    "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SHRIRAMFIN",
    "SBIN", "SUNPHARMA", "TCS", "TATACONSUM", "TATAMOTORS",
    "TATASTEEL", "TECHM", "TITAN", "ULTRACEMCO", "WIPRO",
]

# Nifty IT stocks (for NIFTYIT Qlib model)
NIFTY_IT = [
    "TCS", "INFY", "HCLTECH", "WIPRO", "TECHM",
    "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "OFSS",
]

# Nifty Bank stocks (for BANKING Qlib model)
NIFTY_BANK = [
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN",
    "INDUSINDBK", "BANDHANBNK", "FEDERALBNK", "IDFCFIRSTB", "AUBANK",
    "PNB", "BANKBARODA",
]

# Nifty Pharma stocks (for PHARMA Qlib model)
NIFTY_PHARMA = [
    "SUNPHARMA", "DRREDDY", "CIPLA", "DIVISLAB", "APOLLOHOSP",
    "TORNTPHARM", "AUROPHARMA", "ALKEM", "LUPIN", "BIOCON",
]

# ETF universe (for ETFAgent)
ETF_UNIVERSE = [
    "NIFTYBEES",   # Nifty 50 ETF
    "BANKBEES",    # Nifty Bank ETF
    "GOLDBEES",    # Gold ETF
    "LIQUIDBEES",  # Liquid ETF (cash equivalent)
    "ITBEES",      # IT sector ETF
    "PHARMABEES",  # Pharma ETF
    "JUNIORBEES",  # Nifty Next 50 ETF
    "SETFNN50",    # Nifty Next 50
]

# Market segment to universe mapping
MARKET_UNIVERSES = {
    "NSE500": None,          # Full NSE 500 — fetched dynamically
    "NIFTY50": NIFTY50,
    "NIFTYIT": NIFTY_IT,
    "BANKING": NIFTY_BANK,
    "PHARMA": NIFTY_PHARMA,
    "ETF": ETF_UNIVERSE,
    "SCALPER": NSE_TOP20_LIQUID,
}

# Default flat universe alias (Nifty 50 — used by main.py cycle)
UNIVERSE = [{"symbol": s, "market": "NSE", "segment": "NIFTY50"} for s in NIFTY50]
