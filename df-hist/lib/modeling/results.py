import logging
import os

import pandas as pd
from fantasy_py import dt_to_filename_str

_EVAL_COL_ORDER = [
    "Sport",
    "Service",
    "Type",
    "Style",
    "Target",
    "R2",
    "RMSE",
    "MAE",
    "ModelType",
    "Date",
    "Params",
]

_LOGGER = logging.getLogger(__name__)


def log_eval_results(eval_results: list[dict], name: str, csv_folder):
    """
    write all evaluation results to csv file in the temp folder and return the dataframe
    also write file(s) that describe the final model(s)
    """
    if len(eval_results) == 0:
        _LOGGER.warning("No evaluation results to save")
        return None

    df = pd.DataFrame(eval_results)[_EVAL_COL_ORDER].sort_values(
        ["Sport", "Service", "Type", "Style", "Target", "ModelType"]
    )
    df.Service = df.Service.fillna("multi")
    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)
    results_filepath = os.path.join(csv_folder, name + ".csv")
    df.to_csv(results_filepath, index=False)

    _LOGGER.info("Evaluation results written to '%s'", results_filepath)
    return df


def show_eval_results(eval_results, failed_models, dest_path):
    print(
        f"{len(eval_results) + len(failed_models)} models evaluated, "
        f"{len(failed_models)} failed, {len(eval_results)} successful"
    )
    for n, failure in enumerate(failed_models):
        print(f"failure #{n + 1}: {failure[0]}\n\tcause='{failure[1]['cause']}'")

    if len(eval_results):
        eval_results_df = log_eval_results(
            eval_results, f"all_eval_results-{dt_to_filename_str()}", dest_path
        )

        print(f"{len(eval_results)} successfully serialized models")
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.max_colwidth",
            None,
        ):
            print(eval_results_df.to_csv(index=False, sep="\t"))
