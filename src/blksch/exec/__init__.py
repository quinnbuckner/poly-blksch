"""Track C — execution, ledger, paper engine, dashboard.

Do NOT ``from blksch.exec import *`` — ``exec`` shadows the builtin inside
this package, which is fine as a dotted name but import-star would shadow
the builtin in the caller's namespace.
"""

from .clob_client import CLOBConfig, make_clob_client
from .dashboard import DashboardContext, FlaskDashboard, RichDashboard
from .ledger import Ledger, PnLSnapshot, reconcile
from .order_router import OrderRouter, RouterConfig
from .paper_engine import PaperEngine, PaperEngineConfig, PaperEngineState

__all__ = [
    "CLOBConfig",
    "DashboardContext",
    "FlaskDashboard",
    "Ledger",
    "OrderRouter",
    "PaperEngine",
    "PaperEngineConfig",
    "PaperEngineState",
    "PnLSnapshot",
    "RichDashboard",
    "RouterConfig",
    "make_clob_client",
    "reconcile",
]
