"""
Data processing components for fine-tuning.
"""

try:
    from .processors import DatasetProcessor, JSONLProcessor, AlpacaProcessor
    PROCESSORS_AVAILABLE = True
except ImportError:
    DatasetProcessor = None
    JSONLProcessor = None
    AlpacaProcessor = None
    PROCESSORS_AVAILABLE = False

__all__ = []

if PROCESSORS_AVAILABLE:
    __all__.extend(["DatasetProcessor", "JSONLProcessor", "AlpacaProcessor"])