import os
from math import sqrt
from typing import Literal

import joblib
import numpy as np
import pandas as pd
import sklearn
from fantasy_py import FantasyException, UnexpectedValueError, log
from fantasy_py.analysis.backtest.daily_fantasy.winning_score_range import (
    ModelTarget,
    feature_names_from_win_score_model,
)
from flaml import AutoML as FlamlAutoML
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.multioutput import RegressorChain
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate

from .generate_train_test import TrainTestData

_LOGGER = log.get_logger(__name__)

# quantiles to use for quantile regression models
_QUANTILE_TOP = 0.98
_QUANTILE_LWS = 0.8

Framework = Literal["dummy", "flaml", "ridge", "qreg", "qgbr", "regchain_tree", "regchain_ridge"]
"""
ml framework

dummy - regressor based on a dummy model
flaml - automl flaml model
ridge - ridge regressor
qreg - quantile regressor
qgbr - gradient boosted tree regressor w/quantile loss
regchain_[tree|ridge] - regression chain using decision tree and other regressors,
    first a regressor for top winning score, then use its output as a feature for training
    the low winning score. Only applicable to lws+top
"""


ExistingModelMode = Literal["reuse", "overwrite", "fail"]
"""action to take if a model file already exists"""


def _error_report(
    model,
    target: ModelTarget,
    X_test,
    y_test_fit_data,
    slate_ids,
    desc: str,
    show_results,
    eval_results_path,
) -> dict:
    """
    display the error report for the model, also return a dict with the scores
    """
    if (X_test.columns != feature_names_from_win_score_model(model)).any():
        # find first index of mismatch
        for i, (col, expected_feature) in enumerate(
            zip(X_test.columns, feature_names_from_win_score_model(model))
        ):
            if col == expected_feature:
                continue
            raise UnexpectedValueError(
                f"Invalid input data for model. Columns don't match. First mismatch is col={i + 1}. {expected_feature=} input_col='{col}'"
            )
        raise NotImplementedError("should not get here")
    predictions_raw = model.predict(X_test)

    if target.is_log:
        predictions = np.sign(predictions_raw) * np.expm1(np.abs(predictions_raw))
        y_test = np.sign(y_test_fit_data) * np.expm1(np.abs(y_test_fit_data))
    else:
        predictions = predictions_raw
        y_test = y_test_fit_data

    if target.is_optrat_residual:
        rational = X_test["top_rational_lineup_score"].values
        if target.is_combined:
            predictions = np.column_stack(
                [rational + predictions[:, 0], rational - predictions[:, 1]]
            )
            y_test = np.column_stack([rational + y_test[:, 0], rational - y_test[:, 1]])
        elif target.is_top:
            predictions = rational + predictions
            y_test = rational + y_test
        else:
            predictions = rational - predictions
            y_test = rational - y_test

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions[predictions.columns[0]]

    r2 = round(sklearn.metrics.r2_score(y_test, predictions), 4)
    rmse = round(sqrt(sklearn.metrics.mean_squared_error(y_test, predictions)), 4)
    mae = round(sqrt(sklearn.metrics.mean_absolute_error(y_test, predictions)), 4)
    if target.is_top:
        pinball = round(
            sklearn.metrics.mean_pinball_loss(y_test, predictions, alpha=_QUANTILE_TOP), 4
        )
    elif target.is_lws:
        pinball = round(
            sklearn.metrics.mean_pinball_loss(y_test, predictions, alpha=_QUANTILE_LWS), 4
        )
    elif target.is_combined:
        pinball_losses = {
            "top": sklearn.metrics.mean_pinball_loss(
                y_test[:, 0], predictions[:, 0], alpha=_QUANTILE_TOP
            ),
            "lws": sklearn.metrics.mean_pinball_loss(
                y_test[:, 1], predictions[:, 1], alpha=_QUANTILE_LWS
            ),
        }
        pinball = round((pinball_losses["top"] + pinball_losses["lws"]) / 2, 4)
    else:
        raise NotImplementedError()

    result = {"R2": r2, "RMSE": rmse, "MAE": mae, "pinball": pinball}

    if target.is_combined:
        assert isinstance(y_test, np.ndarray) and y_test.shape[1] == 2
        assert pinball_losses
        truth_top_lws = pd.DataFrame(y_test, columns=["true.top", "true.lws"])
        pred_top_lws = pd.DataFrame(predictions, columns=["pred.top", "pred.lws"])
        for top_lws in ["top", "lws"]:
            truth = truth_top_lws[f"true.{top_lws}"]
            pred = pred_top_lws[f"pred.{top_lws}"]
            result[f"R2.{top_lws}"] = round(sklearn.metrics.r2_score(truth, pred), 4)
            result[f"RMSE.{top_lws}"] = round(
                sqrt(sklearn.metrics.mean_squared_error(truth, pred)), 4
            )
            result[f"MAE.{top_lws}"] = round(
                sqrt(sklearn.metrics.mean_absolute_error(truth, pred)), 4
            )
            result[f"pinball.{top_lws}"] = pinball_losses[top_lws]
    elif target.is_lws:
        result.update({"R2.lws": r2, "RMSE.lws": rmse, "MAE.lws": mae, "pinball.lws": pinball})
    elif target.is_top:
        result.update({"R2.top": r2, "RMSE.top": rmse, "MAE.top": mae, "pinball.top": pinball})
    else:
        raise NotImplementedError()

    if show_results or eval_results_path:
        assert isinstance(predictions, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))

        if isinstance(y_test, np.ndarray) and y_test.shape[1] == 2:
            plot_data_df = pd.concat([truth_top_lws, pred_top_lws], axis=1).assign(
                **{
                    "error.top-lws": np.linalg.norm(y_test - predictions, axis=1),
                    "error.top": truth_top_lws["true.top"] - pred_top_lws["pred.top"],
                    "error.lws": truth_top_lws["true.lws"] - pred_top_lws["pred.lws"],
                    # difference between the true top and lws
                    "true.score-diff": truth_top_lws.diff(axis=1)["true.lws"],
                    # difference between the predicted top and lws
                    "pred.score-diff": pred_top_lws.diff(axis=1)["pred.lws"],
                }
            )
            plot_data_df["error.score-diff"] = (
                plot_data_df["true.score-diff"] - plot_data_df["pred.score-diff"]
            )
        else:
            truth = pd.Series(y_test) if isinstance(y_test, np.ndarray) else y_test
            truth = truth.reset_index(drop=True)
            pred = pd.Series(predictions) if isinstance(predictions, np.ndarray) else predictions
            pred = pred.reset_index(drop=True)
            plot_data_df = pd.concat([truth, pred], axis=1)
            plot_data_df.columns = ["truth", "prediction"]
            plot_data_df["error"] = plot_data_df.prediction - plot_data_df.truth

        plot_data_df.insert(0, "slate_id", slate_ids.reset_index(drop=True))
        if eval_results_path:
            predictions_filename = os.path.join(eval_results_path, desc + ".prediction.csv")
            with open(predictions_filename, "w") as f_:
                plot_data_df.to_csv(f_, index=False)

        if show_results:
            print(f"""

**** Error Report for {desc} ****
{result}

{tabulate(plot_data_df, showindex=False, headers="keys")}
""")

    return result


class FitError(FantasyException):
    """raised if an exception is caught during fit"""


def _fit_model(
    X_train,
    y_train,
    framework: Framework,
    random_state,
    target: ModelTarget,
    model_params: dict,
    model_filepath: str,
):
    if model_params is None:
        model_params = {}
    if framework == "dummy":
        modeler = DummyRegressor(**model_params)
    elif framework.startswith("regchain"):
        if framework == "regchain_tree":
            base_estimator = DecisionTreeRegressor(random_state=random_state, **model_params)
        elif framework == "regchain_ridge":
            base_estimator = Ridge(random_state=random_state, **model_params)
        else:
            raise NotImplementedError()
        # Since the order is 0, 1 and reg chain should be top-score -> lowest score
        # make sure that the target vector is (top-score, low-score)
        modeler = RegressorChain(base_estimator, order=[0, 1])
    elif framework == "ridge":
        modeler = Ridge(random_state=random_state, **model_params)
    elif framework == "flaml":
        modeler = FlamlAutoML(**model_params)
    elif framework == "qgbr":
        if target.is_combined:
            raise UnexpectedValueError(f"quantile not defined for {target=}")
        quantile = _QUANTILE_TOP if target.is_top else _QUANTILE_LWS
        modeler = GradientBoostingRegressor(loss="quantile", alpha=quantile, **model_params)
        if X_train.isna().any().any():
            na_rows = X_train.isna().any(axis=1)
            X_train = X_train[~na_rows]
            y_train = y_train[~na_rows]
    elif framework == "qreg":
        if target.is_combined:
            raise UnexpectedValueError(f"quantile not defined for {target=}")
        quantile = _QUANTILE_TOP if target.is_top else _QUANTILE_LWS
        modeler = QuantileRegressor(quantile=quantile, **model_params)
        if X_train.isna().any().any():
            na_rows = X_train.isna().any(axis=1)
            X_train = X_train[~na_rows]
            y_train = y_train[~na_rows]
    else:
        raise NotImplementedError(f"framework '{framework}' not supported")

    try:
        modeler.fit(X_train, y_train)
        retry = False
    except (AttributeError, RuntimeError) as ex:
        if framework != "flaml" or str(ex) not in (
            "'DummyProcess' object has no attribute 'terminate'",
            "can't start new thread",
        ):
            raise
        _LOGGER.warning("retriable fitting failure with flaml model, lets try this it again...")
        retry = True

    if retry:
        assert framework == "flaml"
        modeler = FlamlAutoML(**(model_params or {}))
        modeler.fit(X_train, y_train)

    # verify that feature names are defined in the model
    feature_names_from_win_score_model(modeler)

    _LOGGER.info("writing model to pickled file '%s'", model_filepath)
    joblib.dump(modeler, model_filepath)
    return modeler


def create_model(
    model_desc: str,
    model_dir: str,
    tt_data: TrainTestData,
    y_train,
    y_test,
    target: ModelTarget,
    framework: Framework,
    random_state=1,
    mode: ExistingModelMode = "fail",
    eval_results_path=None,
    **model_params,
):
    """
    create the model

    X_train, y_train - if not None then train the model
    X_test, y_test - if not None then score
    model_desc - model description used for filename and logging
    model_params - used when creating the model object
    returns - dict containing model, fit_params and evaluation results
    """
    model_filepath = os.path.join(model_dir, model_desc + ".pkl")

    if (file_exists := os.path.isfile(model_filepath)) and mode == "fail":
        raise FileExistsError(f"In 'fail' mode, and model exists at '{model_filepath}'")

    if file_exists and mode == "reuse":
        _LOGGER.info("Reusing model at '%s'", model_filepath)
        model = joblib.load(model_filepath)
    else:
        model = _fit_model(
            tt_data.X_train, y_train, framework, random_state, target, model_params, model_filepath
        )

    eval_results = _error_report(
        model,
        target,
        tt_data.X_test,
        y_test,
        tt_data.test_slate_ids,
        model_desc,
        True,
        eval_results_path,
    )

    return {"model": model, "eval_result": eval_results}
