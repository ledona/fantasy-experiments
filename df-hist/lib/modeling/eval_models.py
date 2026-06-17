from typing import Literal

import numpy as np
from fantasy_py import DataNotAvailableException, DFSContestStyle, log, now
from tqdm import tqdm

from .generate_train_test import ModelFeatures, generate_train_test, load_csv
from .model import ExistingModelMode, FitError, Framework, create_model

_LOGGER = log.get_logger(__name__)

ModelTarget = Literal["top", "lws", "lws+top"]
"""
models targets for fitting/evaluation :
'top' - All available input data predicting the top score
'lws' - All available input data predicting last winning score
'lws+top' - All available input data predicting both lws and top score in 1 shot
"""


def evaluate_models(
    sport,
    style: DFSContestStyle,
    contest_type,
    framework: Framework,
    model_params: dict,
    model_type_pbar: tqdm,
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
            model_type_pbar.update(len(final_model_targets))

            failures += [
                (error_desc_formatter(target, features), {"cause": "Insufficient data"})
                for target in final_model_targets
            ]
            continue

        (X_train, X_test, y_top_train, y_top_test, y_last_win_train, y_last_win_test) = model_data

        raise NotImplementedError()

        for model_target_group in final_model_targets:
            if model_target_group == "top":
                if framework == "reg_chain":
                    _LOGGER.warning(
                        "Skipping model_target_group=%s. framework=%s is does not support it",
                        model_target_group,
                        framework,
                    )
                    continue
                target = "top-score"
                y_train = y_top_train
                y_test = y_top_test
            elif model_target_group == "lws":
                if framework == "reg_chain":
                    _LOGGER.warning(
                        "Skipping model_target_group=%s. framework=%s is does not support it",
                        model_target_group,
                        framework,
                    )
                    continue
                target = "last-win-score"
                y_train = y_last_win_train
                y_test = y_last_win_test
            elif model_target_group == "lws+top":
                if framework != "reg_chain":
                    _LOGGER.warning(
                        "Skipping model_target_group=%s. framework=%s is does not support it",
                        model_target_group,
                        framework,
                    )
                    continue
                target = "top+lw-score"
                y_train = np.column_stack((y_last_win_train, y_top_train))
                y_test = np.column_stack((y_last_win_test, y_top_test))
            else:
                raise NotImplementedError(f"don't know how to train {model_target_group=}")

            model_desc = f"{model_desc_pre}-{target}-{framework}"

            _LOGGER.info("training model=%s params=%s", model_desc, model_params)
            model_type_pbar.set_postfix_str(model_desc)

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
                    model_target_group,
                    framework,
                    ex,
                )
                continue

            models[model_desc] = cam_result["model"]
            model_type_pbar.update()

            finalized_results = cam_result["eval_result"].copy()
            finalized_results["Target"] = target
            finalized_results["Params"] = model_params.copy()
            finalized_results.update(shared_results_dict)

            eval_results.append(finalized_results)

    return models, eval_results, failures
