import os
from math import sqrt
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from fantasy_py import DataNotAvailableException, FantasyException, log
from fantasy_py.analysis.backtest.daily_fantasy.winning_score_range import (
    feature_names_from_win_score_model,
)
from sklearn.dummy import DummyRegressor
from sklearn.multioutput import RegressorChain
from sklearn.tree import DecisionTreeRegressor
from tpot2 import TPOTRegressor

_LOGGER = log.get_logger(__name__)

Framework = Literal["dummy", "tpot", "reg_chain"]
"""ml framework"""

ModelTarget = Literal["top-score", "last-win-score", "top+lw-score"]
"""possible model targets"""

ExistingModelMode = Literal["reuse", "overwrite", "fail"]
"""action to take if a model file already exists"""


def _error_report(model, X_test, y_test, desc: str, show_results, eval_results_path) -> dict:
    """
    display the error report for the model, also return a dict with the scores
    """
    predictions = model.predict(X_test)

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions[predictions.columns[0]]
    r2 = round(sklearn.metrics.r2_score(y_test, predictions), 4)
    rmse = round(sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)), 4)
    mae = round(sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)), 4)

    result = {"R2": r2, "RMSE": rmse, "MAE": mae}

    if isinstance(y_test, np.ndarray) and y_test.shape[1] == 2:
        truth_min_max = pd.DataFrame(y_test, columns=["true.min-win", "true.max-win"])
        pred_min_max = pd.DataFrame(predictions, columns=["pred.min-win", "pred.max-win"])
        for min_max in ["min", "max"]:
            truth = truth_min_max[f"true.{min_max}-win"]
            pred = pred_min_max[f"pred.{min_max}-win"]
            result[f"R2.{min_max}"] = round(sklearn.metrics.r2_score(truth, pred), 4)
            result[f"RSME.{min_max}"] = round(
                sqrt(sklearn.metrics.mean_squared_error(truth, pred)), 4
            )
            result[f"MAE.{min_max}"] = round(
                sqrt(sklearn.metrics.mean_absolute_error(truth, pred)), 4
            )

    if show_results or eval_results_path:
        assert isinstance(predictions, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))

        if isinstance(y_test, np.ndarray) and y_test.shape[1] == 2:
            plot_data = pd.concat([truth_min_max, pred_min_max], axis=1).assign(
                **{
                    "error.min-max": np.linalg.norm(y_test - predictions, axis=1),
                    "error.min": truth_min_max["true.min-win"] - pred_min_max["pred.min-win"],
                    "error.max": truth_min_max["true.max-win"] - pred_min_max["pred.max-win"],
                    "true.score-diff": truth_min_max.diff(axis=1)["true.max-win"],
                    "pred.score-diff": pred_min_max.diff(axis=1)["pred.max-win"],
                }
            )
            plot_data["error.score-diff"] = (
                plot_data["true.score-diff"] - plot_data["pred.score-diff"]
            )
        else:
            truth = pd.Series(y_test) if isinstance(y_test, np.ndarray) else y_test
            truth = truth.reset_index(drop=True)
            pred = pd.Series(predictions) if isinstance(predictions, np.ndarray) else predictions
            pred = pred.reset_index(drop=True)
            plot_data = pd.concat([truth, pred], axis=1)
            plot_data.columns = ["truth", "prediction"]
            plot_data["error"] = plot_data.prediction - plot_data.truth

        if eval_results_path:
            predictions_filename = os.path.join(eval_results_path, desc + ".prediction.csv")
            with open(predictions_filename, "w") as f_:
                plot_data.to_csv(f_, index=False)

        if show_results:
            print(f"**** Error Report for {desc} ****: {result}")
            print()
            print(plot_data)

    return result


class FitError(FantasyException):
    """raised if an exception is caught during fit"""


def _fit_model(
    model_desc,
    X_train,
    y_train,
    framework,
    max_train_time,
    random_state,
    automl_params,
    model_filepath,
):
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
        modeler = TPOTRegressor(
            random_state=random_state,
            max_time_mins=max_train_time,
            **automl_params,
        )
        extract_regressor = lambda model_: model_.fitted_pipeline_
    elif framework == "dummy":
        modeler = DummyRegressor()
        extract_regressor = lambda model_: model_
    elif framework == "reg_chain":
        base_estimator = DecisionTreeRegressor()
        modeler = RegressorChain(base_estimator=base_estimator, order=[0, 1])
        extract_regressor = lambda model_: model_
    else:
        raise NotImplementedError(f"framework '{framework}' not supported")

    try:
        modeler.fit(X_train, y_train, **fit_params)
    except ValueError as ex:
        if "n_quantiles" in str(ex) and len(X_train) <= 10:
            raise FitError(
                "n_quantiles related data failure due to insufficient training data "
                ">10 training samples are required. number of training samples "
                f"available was {len(X_train)}"
            ) from ex

    model = extract_regressor(modeler)
    try:
        feature_names_from_win_score_model(model)
    except DataNotAvailableException:
        if model.steps[0][0].startswith("zero"):
            _LOGGER.info(
                "For '%s' adding 'fantasy_features_names' attribute to the "
                "estimator that does not implement feature_names_in_",
                model_desc,
            )
        else:
            _LOGGER.error(
                "Failed to retrieve feature names from model '%s'. "
                "Adding a 'fantasy_features_names' attribute to the estimator. "
                "Attempt to fix this by explicitly identifying the estimator as "
                "not implementing feature_name_in_ or by by figuring out how to "
                "get features from the estimator type",
                model_desc,
            )
        model.fantasy_feature_names = X_train.columns
    _LOGGER.info("writing model to pickled file '%s'", model_filepath)
    joblib.dump(model, model_filepath)
    return model, fit_params


def create_model(
    model_desc: str,
    model_dir: str,
    X_train,
    y_train,
    X_test,
    y_test,
    random_state=1,
    framework: Framework = "tpot",
    max_train_time=None,
    mode: ExistingModelMode = "fail",
    eval_results_path=None,
    **model_params,
):
    """
    create the model

    max_train_time - time to train the model in seconds
    X_train, y_train - if not None then train the model
    X_test, y_test - if not None then score
    model_desc - model description used for filename and logging
    **automl_params - used when creating the model object

    returns - dict containing model, fit_params and evaluation results
    """
    model_filepath = os.path.join(model_dir, model_desc + ".pkl")

    if (file_exists := os.path.isfile(model_filepath)) and mode == "fail":
        raise FileExistsError(f"In 'fail' mode, and model exists at '{model_filepath}'")

    if file_exists and mode == "reuse":
        _LOGGER.info("Reusing model at '%s'", model_filepath)
        model = joblib.load(model_filepath)
        fit_params = None
    else:
        model, fit_params = _fit_model(
            model_desc,
            X_train,
            y_train,
            framework,
            max_train_time,
            random_state,
            model_params,
            model_filepath,
        )

    eval_results = _error_report(model, X_test, y_test, model_desc, True, eval_results_path)

    return {"model": model, "fit_params": fit_params, "eval_result": eval_results}
