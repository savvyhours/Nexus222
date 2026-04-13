"""
NEXUS-II — core.sentiment
Sentiment pipeline: FinBERT scoring → Haiku enrichment → sector aggregation → Supabase store.
"""
from core.sentiment.finbert_engine import FinBERTEngine, SentimentResult
from core.sentiment.llm_enricher import EnrichedMeta, LLMEnricher
from core.sentiment.sector_aggregator import AggregatedScore, SectorAggregator
from core.sentiment.sentiment_store import SentimentRow, SentimentStore

__all__ = [
    "FinBERTEngine",
    "SentimentResult",
    "LLMEnricher",
    "EnrichedMeta",
    "SectorAggregator",
    "AggregatedScore",
    "SentimentStore",
    "SentimentRow",
]
