import os
from typing import cast

import pandas as pd
from fantasy_py import dt_to_filename_str, log
from tabulate import tabulate

_LOGGER = log.get_logger(__name__)


def _log_eval_results(eval_results: list[dict], dt_str: str, csv_folder):
    """
    Write all evaluation results to csv file in the output folder
    and return the dataframe.
    """
    assert eval_results

    df = pd.DataFrame(eval_results)

    eval_cols = ["Sport", "Service", "Type", "Style", "Target", "Features", "Framework", "Date"]

    for eval_stat in ["R2", "RMSE", "MAE"]:
        eval_cols += [col for col in df if cast(str, col).startswith(eval_stat)]

    eval_cols.append("Params")

    assert not df.Service.isna().any(), "all service values should be defined"
    # df.Service = df.Service.fillna("multi")
    df = df[eval_cols].sort_values(
        ["Sport", "Service", "Type", "Style", "Target", "Features", "Framework", "Date"]
    )
    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)

    results_filepath = os.path.join(csv_folder, f"all_eval_results-{dt_str}.csv")
    df.to_csv(results_filepath, index=False)

    _LOGGER.success("Evaluation results written to '%s'", results_filepath)
    return df


def record_results(eval_results, failed_models, dest_path):
    """print the evaluation results and save them to dest_path"""
    print(
        f"{len(eval_results) + len(failed_models)} models evaluated, "
        f"{len(failed_models)} failed, {len(eval_results)} successful"
    )
    for n, failure in enumerate(failed_models):
        print(f"failure #{n + 1}: {failure[0]}\n\tcause='{failure[1]['cause']}'")

    print(f"{len(eval_results)} successfully serialized models")

    if len(eval_results) == 0:
        return

    dt_str = dt_to_filename_str()
    eval_results_df = _log_eval_results(eval_results, dt_str, dest_path)

    print()
    print(tabulate(eval_results_df, showindex=False, headers="keys"))
