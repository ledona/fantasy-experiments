from typing import Literal

import numpy as np
from fantasy_py import DataNotAvailableException, DFSContestStyle, log, now
from tqdm import tqdm

from .generate_train_test import ModelFeatures, generate_train_test, load_csv
from .model import ExistingModelMode, FitError, Framework, create_model

_LOGGER = log.get_logger(__name__)

ModelTarget = Literal["top", "lws", "top+lws"]
"""
top - top winning score
lws - last winning score
top+lws - single model that predicts for both top and last winning score in 1 shot
"""


def evaluate_models(
    sport,
    style: DFSContestStyle,
    contest_type,
    framework: Framework,
    model_params: dict,
    data_folder="data",
    eval_results_path: str | None = None,
    model_features: set[ModelFeatures] | None = None,
    model_targets: set[ModelTarget] | None = None,
    model_folder="models",
    service=None,
    mode: ExistingModelMode = "fail",
):
    """
    model_targets: set of the models to test. if None then all models \
        are tested for all targets.
    model_features: features to use to train models. If None then all\
        feature sets will be attempted
    eval_results_path: path to write evuation predictions and truth

    returns tuple of (models, evaluation results, failed models)
    """
    service_name = service or "multi"
    models = {}
    eval_results = []
    shared_results_dict = {
        "Sport": sport,
        "Service": service,
        "Style": style.name,
        "Type": contest_type.TYPE_NAME,
        "Framework": framework,
        "Date": now().strftime("%Y%m%d"),
    }
    final_model_targets: set[ModelTarget] = (
        set(ModelTarget.__args__) if model_targets is None else model_targets
    )
    model_desc_pre = "-".join([sport, service_name, style.name, contest_type.TYPE_NAME, framework])
    final_model_feature_sets = model_features or ModelFeatures.__args__

    def error_desc_formatter(targ, feats):
        return model_desc_pre + f"-{targ}-{feats}"

    try:
        df = load_csv(sport, service, style, contest_type, data_folder=data_folder)
    except DataNotAvailableException as ex:
        _LOGGER.error(
            "Data file(s) required for modeling not found. Skipping %s",
            model_desc_pre,
            exc_info=ex,
        )
        failures = [
            (error_desc_formatter(target, features), {"cause": "No data file found"})
            for target in final_model_targets
            for features in final_model_feature_sets
        ]
        return None, None, failures

    failures = []
    for features in (
        features_pbar := tqdm(
            final_model_feature_sets, desc="Features", disable=len(final_model_feature_sets) == 1
        )
    ):
        features_pbar.set_postfix_str(features)
        model_data = generate_train_test(
            df,
            sport,
            style,
            features,
            random_state=model_params.get("random_state", 0),
            service_as_feature=(service is None),
        )
        if model_data is None or len(model_data[0]) < 5:
            _LOGGER.error(
                "Not enough training data available for %s features:%s! Only found %i training cases. Skipping...",
                model_desc_pre,
                features,
                (len(model_data[0]) if model_data else 0),
            )

            failures += [
                (error_desc_formatter(target, features), {"cause": "Insufficient data"})
                for target in final_model_targets
            ]
            continue

        (X_train, X_test, y_top_train, y_top_test, y_last_win_train, y_last_win_test) = model_data

        for target in (
            target_pbar := tqdm(
                final_model_targets, desc="Targets", disable=len(final_model_targets) == 1
            )
        ):
            target_pbar.set_postfix_str(target)
            if (framework == "reg_chain" and target != "top+lws") or (
                framework == "flaml" and target == "top+lws"
            ):
                _LOGGER.warning(
                    "Skipping model_target=%s. framework=%s does not support it",
                    target,
                    framework,
                )
                continue

            if target == "top":
                y_train = y_top_train
                y_test = y_top_test
            elif target == "lws":
                y_train = y_last_win_train
                y_test = y_last_win_test
            elif target == "top+lws":
                # make sure this is (top, last-winning)
                y_train = np.column_stack((y_top_train, y_last_win_train))
                y_test = np.column_stack((y_top_test, y_last_win_test))
            else:
                raise NotImplementedError(f"don't know how to train {target=}")

            model_desc = f"{model_desc_pre}-t:{target}-f:{features}"

            _LOGGER.info("training model=%s params=%s", model_desc, model_params)

            try:
                cam_result = create_model(
                    model_desc,
                    model_folder,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    framework=framework,
                    mode=mode,
                    eval_results_path=eval_results_path,
                    **model_params,
                )
            except FitError as ex:
                _LOGGER.warning(
                    "Skipping model_target_group=%s. framework=%s due to fitting error. ex=%s",
                    target,
                    framework,
                    ex,
                )
                continue

            models[model_desc] = cam_result["model"]

            finalized_results = {
                **shared_results_dict,
                **cam_result["eval_result"],
                "Target": target,
                "Features": features,
                "Params": model_params.copy(),
            }

            eval_results.append(finalized_results)

    return models, eval_results, failures
