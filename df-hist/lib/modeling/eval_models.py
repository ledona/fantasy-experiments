import logging
from datetime import datetime
from typing import Literal

import pandas as pd

from .automl import ExistingModelMode, Framework, ModelTarget, create_automl_model
from .generate_train_test import generate_train_test, load_csv

_LOGGER = logging.getLogger(__name__)

ModelTargetGroup = Literal[
    "all-top",
    "all-lws",
]
"""
models that can be evaluated :
'all-top' - All available input data predicting the top score
'all-lws' - All available input data predicting last winning score
"""


def _targets_from_models_to_test(models_to_test: ModelTargetGroup) -> list[str]:
    targets = []
    for model_to_test in models_to_test:
        if model_to_test == "all-top":
            targets.append("top-score")
        elif model_to_test == "all-lws":
            targets.append("last-winning-score")
        else:
            raise ValueError(f"Unknown model to test: {model_to_test}")
    return targets


def evaluate_models(
    sport,
    style,
    contest_type,
    framework: Framework,
    automl_params: dict,
    pbar,
    data_folder="data",
    models_to_test: set[ModelTargetGroup] | None = None,
    model_folder="models",
    service=None,
    mode: ExistingModelMode = "fail",
) -> tuple[dict, list, list[tuple[str, str]]]:
    """
    models_to_test - set/list of the models to test. if None then all models tested.
        possible models are
        if service is None then use all service data and add a service feature
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
        "ModelType": framework,
        "Date": datetime.now().strftime("%Y%m%d"),
    }
    final_models_to_test = {"all-top", "all-lws"} if models_to_test is None else models_to_test
    model_desc_pre = "-".join([sport, service_name, style.name, contest_type.NAME])

    try:
        df = load_csv(sport, service, style, contest_type, data_folder=data_folder)
    except FileNotFoundError as ex:
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
        random_state=automl_params.get("random_state", 0),
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

    (
        X_train,
        X_test,
        y_top_train,
        y_top_test,
        y_last_win_train,
        y_last_win_test,
    ) = model_data

    model_ys: list[tuple[ModelTarget, pd.Series, pd.Series]] = []
    if "all-top" in final_models_to_test:
        model_ys.append(("top-score", y_top_train, y_top_test))
    if "all-lws" in final_models_to_test:
        model_ys.append(("last-win-score", y_last_win_train, y_last_win_test))

    if len(model_ys) == 0:
        raise ValueError(f"No models to test: {final_models_to_test}")

    # models for top and last winning score
    for target, y_train, y_test in model_ys:
        model_desc = f"{model_desc_pre}-{target}-{framework}"

        _LOGGER.info("training model=%s", model_desc)
        pbar.set_postfix_str(model_desc)

        cam_result = create_automl_model(
            model_desc,
            model_folder,
            X_train,
            y_train,
            X_test,
            y_test,
            framework=framework,
            mode=mode,
            **automl_params,
        )

        models[model_desc] = cam_result["model"]
        pbar.update()

        finalized_results = cam_result["eval_result"].copy()
        finalized_results["Target"] = target
        finalized_results["Params"] = automl_params.copy()
        finalized_results.update(shared_results_dict)

        eval_results.append(finalized_results)

    return models, eval_results, None
