"""
NEXUS-II — core.execution
Order lifecycle: OrderManager routes to DhanExecutor (LIVE) or PaperTrader (PAPER_TRADE).
"""
from core.execution.order_manager import (
    ExecutionReceipt,
    OrderManager,
    OrderRequest,
    OrderStatus,
    OrderType,
    ProductType,
    TransactionType,
)
from core.execution.paper_trader import PaperTrader, SimPosition
from core.execution.dhan_executor import DhanExecutor

__all__ = [
    "OrderManager",
    "OrderRequest",
    "ExecutionReceipt",
    "OrderStatus",
    "OrderType",
    "ProductType",
    "TransactionType",
    "DhanExecutor",
    "PaperTrader",
    "SimPosition",
]
