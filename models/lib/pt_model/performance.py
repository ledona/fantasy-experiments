import json
import os
from typing import Literal

import pandas as pd
import tqdm
from fantasy_py import log, InvalidArgumentsException
from fantasy_py.inference import PerformanceDict, PTPredictModel
from sklearn import metrics

from .cfg import TrainingConfiguration
from .train_test import load_data

_LOGGER = log.get_logger(__name__)

PerformanceOperation = Literal["calc", "update", "repair", "test"]
"""supported performance operations"""

_EXPECTED_PERFORMANCE_KEYS = set(PerformanceDict.__annotations__.keys())


def _update_models(perf_recs: dict[str, PerformanceDict]):
    """
    for each model that will be updated
    1. rename the original file to [name].model.backup
    2. load the json and update it with the new performance information
    3. save to the original filename
    """
    for model_filepath, perf in perf_recs.items():
        _LOGGER.info("Updating '%s' with new performance data", model_filepath)

        with open(model_filepath, "r") as mf:
            model_dict = json.load(mf)
        assert "performance" in model_dict["meta_extra"]
        model_dict["meta_extra"]["performance"] = perf

        backup_filepath = model_filepath + ".backup"
        if os.path.isfile(backup_filepath):
            raise FileExistsError(
                f"Could not create a backup for '{model_filepath}' "
                "because the backup file already exists"
            )
        os.rename(model_filepath, backup_filepath)

        with open(model_filepath, "w") as mf:
            json.dump(model_dict, mf, indent="\t")


def _fix_input_col_names(model: PTPredictModel, X_test: pd.DataFrame, X_val: pd.DataFrame):
    if (
        model._input_cols is not None
        and "pos_<NA>" in model._input_cols
        and "pos_<NA>" not in X_test
        and "pos_None" in X_test
    ):
        _LOGGER.info(" remapping input column 'pos_None' -> 'pos_<NA>'")
        return X_test.rename(columns={"pos_None": "pos_<NA>"}), X_val.rename(
            columns={"pos_None": "pos_<NA>"}
        )

    return X_test, X_val


def performance_calc(
    operation: PerformanceOperation,
    model_filepaths: list[str],
    cfg: TrainingConfiguration | None,
    data_dir: str | None,
):
    """performance recalculation"""
    performance_records: dict[str, PerformanceDict] = {}
    test_models = operation in ("repair", "test")
    update_models = operation in ("repair", "update")
    models_w_invalid_perf = []

    for model_filepath in tqdm.tqdm(model_filepaths):
        _LOGGER.info("Processing '%s'", model_filepath)
        model = PTPredictModel.load(model_filepath)

        if test_models:
            invalid = model.performance is None or (
                set(model.performance.keys()) != _EXPECTED_PERFORMANCE_KEYS
            )
            if invalid:
                _LOGGER.warning("'%s' performance is not valid", model_filepath)
                models_w_invalid_perf.append(model_filepath)
            else:
                _LOGGER.success("'%s' performance is valid", model_filepath)
            if not invalid or operation == "test":
                continue

        if cfg is None or data_dir is None:
            raise InvalidArgumentsException(
                "Performance cannot be calculated without data-dir and model config file"
            )

        params = cfg.get_params(model.name)

        data_filepath = params["data_filename"]
        if data_dir is not None:
            data_filepath = os.path.join(data_dir, data_filepath)

        tt_data = load_data(
            data_filepath,
            (model.target.type, model.target.name),
            params["validation_season"],
            params["seed"],
            include_position=params["include_pos"],
            col_drop_filters=params["cols_to_drop"],
            filtering_query=params["filtering_query"],
            skip_data_reports=True,
        )[1]

        (X_test, y_test, X_val, y_val) = tt_data[2:]

        X_test, X_val = _fix_input_col_names(model, X_test, X_val)

        _LOGGER.info("   '%s'-predicting for test", model.name)
        imputed_x_test = model._impute_missing(X_test)
        y_hat = model.model_predict(imputed_x_test)
        _LOGGER.info("   '%s'-predicting for validation", model.name)
        imputed_x_val = model._impute_missing(X_val)
        y_hat_val = model.model_predict(imputed_x_val)

        perf_rec: PerformanceDict = {
            "r2_test": float(metrics.r2_score(y_test, y_hat)),
            "mae_test": float(metrics.mean_absolute_error(y_test, y_hat)),
            "r2_val": float(metrics.r2_score(y_val, y_hat_val)),
            "mae_val": float(metrics.mean_absolute_error(y_val, y_hat_val)),
            "season_val": params["validation_season"],
        }

        _LOGGER.info(
            "   '%s'-test       r2=%g mae=%g",
            model.name,
            round(perf_rec["r2_test"], 6),
            round(perf_rec["mae_test"], 6),
        )
        _LOGGER.info(
            "   '%s'-validation r2=%g mae=%g",
            model.name,
            round(perf_rec["r2_val"], 6),
            round(perf_rec["mae_val"], 6),
        )
        performance_records[model_filepath] = perf_rec

    if test_models:
        if len(models_w_invalid_perf) > 0:
            _LOGGER.info(
                "Test results - %i of %i model(s) have invalid performance data",
                len(models_w_invalid_perf),
                len(model_filepaths),
            )
            print(f"\nINVALID MODEL FILES (n={len(models_w_invalid_perf)})")
            for model_fp in sorted(models_w_invalid_perf):
                print(model_fp)
        else:
            _LOGGER.info("Test results - All model files are valid")
            return

        if operation == "test" or len(models_w_invalid_perf) == 0:
            return

    assert len(performance_records) > 0, "There should have been some calculations"

    df = pd.DataFrame.from_dict(
        performance_records,
        orient="index",
    )
    df.index = pd.Index([os.path.basename(fp) for fp in performance_records], name="model")

    with pd.option_context(
        "display.max_rows",
        None,
        "display.max_colwidth",
        100,
    ):
        print(df.to_string())

    if operation == "calc":
        return
    assert update_models, "Should only get here if we are updating models"
    _update_models(performance_records)
