from typing import Literal

import numpy as np
from fantasy_py import DataNotAvailableException, log, now

from .generate_train_test import generate_train_test, load_csv
from .model import ExistingModelMode, FitError, Framework, create_model

_LOGGER = log.get_logger(__name__)

ModelTargetGroup = Literal["all-top", "all-lws", "all-lws+top"]
"""
models that can be evaluated :
'all-top' - All available input data predicting the top score
'all-lws' - All available input data predicting last winning score
'all-lws+top'   - All available input data predicting both lws and top score in 1 shot
"""


def _targets_from_models_to_test(models_to_test: set[ModelTargetGroup]) -> list[str]:
    targets = []
    for model_to_test in models_to_test:
        if model_to_test == "all-top":
            targets.append("top-score")
        elif model_to_test == "all-lws":
            targets.append("last-winning-score")
        elif model_to_test == "all-lws+top":
            targets.append("top+lw-score")
        else:
            raise ValueError(f"Unknown model to test: {model_to_test}")
    return targets


def evaluate_models(
    sport,
    style,
    contest_type,
    framework: Framework,
    model_params: dict,
    pbar,
    data_folder="data",
    eval_results_path: str | None = None,
    models_to_test: set[ModelTargetGroup] | None = None,
    model_folder="models",
    service=None,
    mode: ExistingModelMode = "fail",
) -> tuple[dict | None, list | None, list[tuple[str, str]] | None]:
    """
    models_to_test: set/list of the models to test. if None then all models \
        tested. Possible models are if service is None then use all service \
        data and add a service feature
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
        "Type": contest_type.NAME,
        "Framework": framework,
        "Date": now().strftime("%Y%m%d"),
    }
    final_models_to_test: set[ModelTargetGroup] = (
        set(ModelTargetGroup.__args__) if models_to_test is None else models_to_test
    )
    model_desc_pre = "-".join([sport, service_name, style.name, contest_type.NAME])

    try:
        df = load_csv(sport, service, style, contest_type, data_folder=data_folder)
    except DataNotAvailableException as ex:
        _LOGGER.error(
            "Data file(s) required for modeling not found. Skipping %s",
            model_desc_pre,
            exc_info=ex,
        )
        pbar.update(len(final_models_to_test))
        failed_models = [
            (
                f"{sport}-{service_name}-{style.name}-{contest_type.NAME}-{target}-{framework}",
                {"cause": "No data file found"},
            )
            for target in _targets_from_models_to_test(final_models_to_test)
        ]
        return None, None, failed_models

    model_data = generate_train_test(
        df,
        model_cols=None,
        random_state=model_params.get("random_state", 0),
        service_as_feature=(service is None),
    )

    if model_data is None or len(model_data[0]) < 5:
        _LOGGER.error(
            "Not enough training data available! Only found %i training cases. Skipping %s",
            (len(model_data[0]) if model_data else 0),
            model_desc_pre,
        )
        pbar.update(len(final_models_to_test))
        failed_models = [
            (f"{model_desc_pre}-{target}-{framework}", {"cause": "Insufficient data"})
            for target in _targets_from_models_to_test(final_models_to_test)
        ]
        return None, None, failed_models

    (X_train, X_test, y_top_train, y_top_test, y_last_win_train, y_last_win_test) = model_data

    if len(final_models_to_test) == 0:
        raise ValueError(f"No models to test: {final_models_to_test}")

    for model_target_group in final_models_to_test:
        if model_target_group == "all-top":
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
        elif model_target_group == "all-lws":
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
        elif model_target_group == "all-lws+top":
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
        pbar.set_postfix_str(model_desc)

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
        pbar.update()

        finalized_results = cam_result["eval_result"].copy()
        finalized_results["Target"] = target
        finalized_results["Params"] = model_params.copy()
        finalized_results.update(shared_results_dict)

        eval_results.append(finalized_results)

    return models, eval_results, None
