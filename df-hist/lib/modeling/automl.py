import logging
import os
from math import floor, sqrt
from typing import Literal

import dask
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyRegressor
from tpot import TPOTRegressor

dask.config.set(scheduler="processes", num_workers=floor(os.cpu_count() * 0.75))

_LOGGER = logging.getLogger(__name__)

# Frameworks = Literal["skautoml", "tpot"]
Framework = Literal["dummy", "tpot"]
"""automl framework"""

ModelTarget = Literal["top-score", "last-win-score"]
"""possible model targets"""


def _error_report(
    model,
    X_test,
    y_test,
    desc: str = None,
    show_results=True,
) -> dict:
    """
    display the error report for the model, also return a dict with the scores
    """
    predictions = model.predict(X_test)

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions[predictions.columns[0]]
    r2 = round(sklearn.metrics.r2_score(y_test, predictions), 4)
    rmse = round(sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)), 4)
    mae = round(sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)), 4)

    result = {
        "R2": r2,
        "RMSE": rmse,
        "MAE": mae,
    }
    if show_results:
        _LOGGER.info("**** Error Report for %s ****: %s", desc, result)
        assert isinstance(predictions, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))

        truth = pd.Series(y_test) if isinstance(y_test, np.ndarray) else y_test
        truth = y_test.reset_index(drop=True)
        pred = pd.Series(predictions) if isinstance(predictions, np.ndarray) else predictions
        pred = pred.reset_index(drop=True)

        plot_data = pd.concat([truth, pred], axis=1)
        plot_data.columns = ["truth", "prediction"]
        plot_data["error"] = plot_data.prediction - plot_data.truth
        print(plot_data)

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(f"{desc or 'unknown model'} : {r2=} {rmse=} {mae=}")
        for ax in axs:
            ax.axis("equal")

        min_v = min(plot_data.truth.min(), plot_data.prediction.min())
        max_v = max(plot_data.truth.max(), plot_data.prediction.max())

        axs[0].plot((min_v, max_v), (min_v, max_v), "-g", linewidth=1)
        plot_data.plot(kind="scatter", x="truth", y="prediction", ax=axs[0])

        axs[1].yaxis.set_label_position("right")
        axs[1].plot((min_v, max_v), (0, 0), "-g", linewidth=1)
        plot_data.plot(kind="scatter", x="truth", y="error", ax=axs[1])

    return result, predictions


def create_automl_model(
    model_desc: str,
    model_dir: str,
    random_state=1,
    framework: Framework = "tpot",
    max_train_time=None,
    X_train=None,
    y_train=None,
    X_test=None,
    y_test=None,
    **automl_params,
):
    """
    create the model

    max_train_time - time to train the model in seconds
    X_train, y_train - if not None then train the model
    X_test, y_test - if not None then score
    model_desc - required if X_test and y_test are not None
    **automl_params - used when creating the model object

    returns - dict containing model, fit_params and evaluation results
    """
    fit_params = {}
    if framework == "tpot":
        if max_train_time is not None:
            if (rem := max_train_time % 60) != 0:
                new_train_time = max_train_time + 60 - rem
                _LOGGER.warning(
                    "TPot requires a training time in minutes. Rounding requested "
                    "train time from %i up to %i seconds",
                    max_train_time,
                    new_train_time,
                )
                max_train_time = new_train_time
            max_train_time /= 60
        model = TPOTRegressor(
            random_state=random_state,
            max_time_mins=max_train_time,
            **automl_params,
        )
        extract_regressor = lambda model_: model_.fitted_pipeline_
    elif framework == "dummy":
        model = DummyRegressor()
        extract_regressor = lambda model_: model_
    else:
        raise NotImplementedError(f"framework '{framework}' not supported")

    eval_results = None
    if X_train is not None and y_train is not None:
        Xtr = X_train
        Xte = X_test
        model.fit(Xtr, y_train, **fit_params)
        if X_test is not None and y_test is not None:
            eval_results, predictions = _error_report(model, Xte, y_test, desc=model_desc)

    model_filepath = os.path.join(model_dir, model_desc + ".pkl")
    _LOGGER.info("writing model to pickled file '%s'", model_filepath)

    joblib.dump(extract_regressor(model), model_filepath)

    return {
        "model": model,
        "fit_params": fit_params,
        "eval_result": eval_results,
        "predictions": predictions,
    }
