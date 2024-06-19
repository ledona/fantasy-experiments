"""package containing code that for player/team models"""

from .cfg import (
    DEFAULT_ALGORITHM,
    TRAINING_PARAM_DEFAULTS,
    TrainingConfiguration,
    _TrainingParamsDict,
)
from .performance import PerformanceOperation, performance_calc
from .train_test import AlgorithmType, ModelFileFoundMode, load_data, model_and_test

__all__ = [
    "TrainingConfiguration",
    "AlgorithmType",
    "load_data",
    "PerformanceOperation",
    "performance_calc",
    "model_and_test",
    "_TrainingParamsDict",
    "TRAINING_PARAM_DEFAULTS",
    "DEFAULT_ALGORITHM",
    "ModelFileFoundMode",
]
