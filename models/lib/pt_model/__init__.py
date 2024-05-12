"""package containing code that for player/team models"""

from .cfg import DEFAULT_ALGORITHM, TrainingConfiguration, _TrainingParamsDict
from .train_test import AlgorithmType, load_data, model_and_test

__all__ = [
    "TrainingConfiguration",
    "AlgorithmType",
    "load_data",
    "model_and_test",
    "_TrainingParamsDict",
    "DEFAULT_ALGORITHM",
]
