from collections.abc import Collection

import numpy as np
from fantasy_py import DataNotAvailableException, DFSContestStyle, log, now
from fantasy_py.analysis.backtest.daily_fantasy import ModelFeatures, ModelTarget, model_filenamer
from tqdm import tqdm

from .generate_train_test import TrainTestData, generate_train_test, load_csv
from .model import ExistingModelMode, FitError, Framework, create_model

_LOGGER = log.get_logger(__name__)


def _get_target_values(target: ModelTarget, tt_data: TrainTestData):
    """returns (training-data, test/eval-data)"""
    if target.is_combined:
        if target.is_optrat_residual:
            top_orr_train = tt_data.y_train_top - tt_data.X_train.top_rational_lineup_score
            lws_orr_train = tt_data.X_train.top_rational_lineup_score - tt_data.y_train_lws
            top_orr_test = tt_data.y_test_top - tt_data.X_test.top_rational_lineup_score
            lws_orr_test = tt_data.X_test.top_rational_lineup_score - tt_data.y_test_lws
            y_train = np.column_stack((top_orr_train, lws_orr_train))
            y_test = np.column_stack((top_orr_test, lws_orr_test))
        else:
            y_train = np.column_stack((tt_data.y_train_top, tt_data.y_train_lws))
            y_test = np.column_stack((tt_data.y_test_top, tt_data.y_test_lws))
    elif target.is_top:
        y_train, y_test = tt_data.y_train_top, tt_data.y_test_top
        if target.is_optrat_residual:
            # residual = truth - optimal rational lineup score
            y_train = y_train - tt_data.X_train.top_rational_lineup_score
            y_test = y_test - tt_data.X_test.top_rational_lineup_score
    elif target.is_lws:
        y_train, y_test = tt_data.y_train_lws, tt_data.y_test_lws
        if target.is_optrat_residual:
            # residual = optimal rational lineup score - truth
            y_train = tt_data.X_train.top_rational_lineup_score - y_train
            y_test = tt_data.X_test.top_rational_lineup_score - y_test
    else:
        raise NotImplementedError(f"don't know how to get base values for {target=}")

    if target.is_log:
        # signed log to handle negative residuals: sign(y) * log1p(|y|)
        return np.sign(y_train) * np.log1p(np.abs(y_train)), np.sign(y_test) * np.log1p(
            np.abs(y_test)
        )

    return y_train, y_test


def evaluate_models(
    sport,
    style: DFSContestStyle,
    contest_type,
    framework: Framework,
    model_params: dict,
    data_folder="data",
    eval_results_path: str | None = None,
    model_features: Collection[ModelFeatures] | None = None,
    model_targets: Collection[ModelTarget] | None = None,
    model_folder="models",
    service: str | None = None,
    mode: ExistingModelMode = "fail",
):
    """
    model_targets: models to test. if None then all models \
        are tested for all targets.
    model_features: features to use to train models. If None then all\
        feature sets will be attempted
    eval_results_path: path to write evuation predictions and truth

    returns tuple of (models, evaluation results, failed models)
    """
    assert service
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
    final_model_targets = sorted(
        ModelTarget.all_instances() if model_targets is None else model_targets
    )
    model_desc_pre = model_filenamer(
        sport=sport, service=service, style=style, contest_type=contest_type, framework=framework
    )
    final_model_feature_sets = sorted(model_features or ModelFeatures.__args__)

    def error_desc_formatter(targ, feats):
        return model_desc_pre + f"-{targ}-{feats}"

    try:
        df = load_csv(sport, service, style, contest_type, data_folder=data_folder)
    except DataNotAvailableException as ex:
        _LOGGER.error(
            "Data required for fitting %s not returned. Skipping...", model_desc_pre, exc_info=ex
        )
        failures = [
            (
                model_filenamer(prefix=model_desc_pre, target=target, features=features),
                {"cause": "No data file found"},
            )
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
        tt_data = generate_train_test(
            df,
            sport,
            style,
            features,
            drop_na_rows=(framework.startswith("regchain") or "ridge" in framework),
            random_state=model_params.get("random_state", 0),
        )
        if tt_data is None or len(tt_data.X_train) < 5:
            _LOGGER.error(
                "Not enough training data available for %s features:%s! Only found %i training cases. Skipping...",
                model_desc_pre,
                features,
                (len(tt_data.X_train) if tt_data else 0),
            )

            failures += [
                (error_desc_formatter(target, features), {"cause": "Insufficient data"})
                for target in final_model_targets
            ]
            continue

        for target in (
            target_pbar := tqdm(
                final_model_targets, desc="Targets", disable=len(final_model_targets) == 1
            )
        ):
            target_pbar.set_postfix_str(str(target))

            # regchain and target.is_combined have to both be true or both be false, otherwise skip
            if framework.startswith("regchain") != target.is_combined:
                _LOGGER.info(
                    "Skipping model_target=%s framework=%s. this combination is not supported",
                    target,
                    framework,
                )
                continue

            y_train, y_test = _get_target_values(target, tt_data)
            model_desc = model_filenamer(prefix=model_desc_pre, target=target, features=features)

            _LOGGER.info("training model=%s params=%s", model_desc, model_params)

            try:
                cam_result = create_model(
                    model_desc,
                    model_folder,
                    tt_data,
                    y_train,
                    y_test,
                    target,
                    framework,
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
                "Target": str(target),
                "Features": features,
                "Params": model_params.copy(),
            }

            eval_results.append(finalized_results)

    return models, eval_results, failures
