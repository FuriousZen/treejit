"""treejit: an inference proxy with memory.

Learns a persistent execution tree from successful agent runs and replays proven
tool-call sequences with zero model calls; the model is only invoked at the frontier.
"""

from .config import Config
from .engine import Result, TreeJIT
from .inline import wrap

__all__ = ["Config", "Result", "TreeJIT", "wrap"]
__version__ = "0.1.0"
