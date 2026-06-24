import os
from math import sqrt
from typing import Literal, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
from fantasy_py import DataNotAvailableException, FantasyException, log
from fantasy_py.analysis.backtest.daily_fantasy.winning_score_range import (
    feature_names_from_win_score_model,
)
from flaml import AutoML as FlamlAutoML
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import RegressorChain
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate

_LOGGER = log.get_logger(__name__)

Framework = Literal["dummy", "flaml", "ridge", "regchain_tree", "regchain_ridge"]
"""
ml framework/method

dummy - regressor based on a dummy model
regchain_tree - regression chain using tree regressors, first a regressor for top winning\
    score, then use its output as a feature in a regressor for the low winning score.\
    Only applicable to lws+top
regchain_ridge - same as regchain_tree but uses a ridge regressor
flaml - automl flaml model
ridge - ridge regressor
"""

ModelTarget = Literal["top", "lws", "top+lws", "top_log", "lws_log", "top+lws_log"]
"""
top - top winning score
lws - last winning score
top+lws - single model that predicts for both top and last winning score in 1 shot
..._log - same as previous but the log of the target score will be the target
"""

ExistingModelMode = Literal["reuse", "overwrite", "fail"]
"""action to take if a model file already exists"""


def _error_report(
    model, target: ModelTarget, X_test, y_test, desc: str, show_results, eval_results_path
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

    result = {"R2": r2, "RMSE": rmse, "MAE": mae}

    if target.startswith("top+lws"):
        assert isinstance(y_test, np.ndarray) and y_test.shape[1] == 2
        truth_top_lws = pd.DataFrame(y_test, columns=["true.top", "true.lws"])
        pred_top_lws = pd.DataFrame(predictions, columns=["pred.top", "pred.lws"])
        for top_lws in ["top", "lws"]:
            truth = truth_top_lws[f"true.{top_lws}"]
            pred = pred_top_lws[f"pred.{top_lws}"]
            result[f"R2.{top_lws}"] = round(sklearn.metrics.r2_score(truth, pred), 4)
            result[f"RSME.{top_lws}"] = round(
                sqrt(sklearn.metrics.mean_squared_error(truth, pred)), 4
            )
            result[f"MAE.{top_lws}"] = round(
                sqrt(sklearn.metrics.mean_absolute_error(truth, pred)), 4
            )
    elif target.startswith("lws"):
        result.update({"R2.lws": r2, "RMSE.lws": rmse, "MAE.lws": mae})
    elif target.startswith("top"):
        result.update({"R2.top": r2, "RMSE.top": rmse, "MAE.top": mae})
    else:
        raise NotImplementedError()

    if show_results or eval_results_path:
        assert isinstance(predictions, (pd.Series, np.ndarray))
        assert isinstance(y_test, (pd.Series, np.ndarray))

        if isinstance(y_test, np.ndarray) and y_test.shape[1] == 2:
            if target.endswith("_log"):
                truth_top_lws_score_df = cast(pd.DataFrame, np.expm1(truth_top_lws))
                pred_top_lws_score_df = cast(pd.DataFrame, np.expm1(pred_top_lws))
                y_test_score = np.expm1(y_test)
                predictions_score = np.expm1(predictions)
            else:
                truth_top_lws_score_df = truth_top_lws
                pred_top_lws_score_df = pred_top_lws
                y_test_score = y_test
                predictions_score = predictions

            plot_data_df = pd.concat(
                [truth_top_lws_score_df, pred_top_lws_score_df], axis=1
            ).assign(
                **{
                    "error.top-lws": np.linalg.norm(y_test_score - predictions_score, axis=1),
                    "error.top": truth_top_lws_score_df["true.top"]
                    - pred_top_lws_score_df["pred.top"],
                    "error.lws": truth_top_lws_score_df["true.lws"]
                    - pred_top_lws_score_df["pred.lws"],
                    # difference between the true top and lws
                    "true.score-diff": truth_top_lws_score_df.diff(axis=1)["true.lws"],
                    # difference between the predicted top and lws
                    "pred.score-diff": pred_top_lws_score_df.diff(axis=1)["pred.lws"],
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
            if target.endswith("_log"):
                plot_data_df = cast(pd.DataFrame, np.expm1(plot_data_df))
            plot_data_df.columns = ["truth", "prediction"]
            plot_data_df["error"] = plot_data_df.prediction - plot_data_df.truth

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
    model_desc,
    X_train,
    y_train,
    framework: Framework,
    random_state,
    model_params,
    model_filepath,
):
    if framework == "dummy":
        modeler = DummyRegressor(**(model_params or {}))
    elif framework.startswith("regchain"):
        if framework == "regchain_tree":
            base_estimator = DecisionTreeRegressor(
                random_state=random_state, **(model_params or {})
            )
        elif framework == "regchain_ridge":
            base_estimator = Ridge(random_state=random_state, **(model_params or {}))
        else:
            raise NotImplementedError()
        # Since the order is 0, 1 and reg chain should be top-score -> lowest score
        # make sure that the target vector is (top-score, low-score)
        modeler = RegressorChain(base_estimator, order=[0, 1])
    elif framework == "ridge":
        modeler = Ridge(random_state=random_state, **(model_params or {}))
    elif framework == "flaml":
        modeler = FlamlAutoML(**(model_params or {}))
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
    X_train,
    y_train,
    X_test,
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
            model_desc,
            X_train,
            y_train,
            framework,
            random_state,
            model_params,
            model_filepath,
        )

    eval_results = _error_report(
        model, target, X_test, y_test, model_desc, True, eval_results_path
    )

    return {"model": model, "eval_result": eval_results}
