"""
Load model eval result csv file, reorganize the data by pairing
complementary (pairable) top and lws models along with the pair's
performance metrics, save the model top and lws model pair evaluation results
in a way that is easier to review than the original eval results AND
if such that it can easily by used/reused to identify the model pairs
that should be used when backtesting"""

import glob
import os
import pathlib
import shlex
from argparse import ArgumentParser
from itertools import chain
from typing import cast

import pandas as pd
from fantasy_py import log
from fantasy_py.analysis.backtest.daily_fantasy import (
    WINSCORE_MODEL_RESULTS_SUBDIR,
    ModelTarget,
    model_filenamer,
)
from tabulate import tabulate
from tqdm import tqdm


def _xo_rate(row: pd.Series, pred_dir_path: str):
    """pandas apply function for calculating crossover rate"""
    lws_top_preds_df: pd.DataFrame
    target_post = ("_log" if row.log else "") + ("_orr" if row.orr else "")

    if "regchain" in row.Framework:
        pred_filename = (
            model_filenamer(
                sport=row.Sport,
                service=row.Service,
                style=row.Style,
                contest_type=row.Type,
                framework=row.Framework,
                target=ModelTarget.from_value(row.combined_target),
                features=row.Features,
            )
            + ".prediction.csv"
        )
        lws_top_preds_df = pd.read_csv(os.path.join(pred_dir_path, pred_filename))
    else:
        preds = {}
        for key in ["top", "lws"]:
            pred_filename = (
                model_filenamer(
                    sport=row.Sport,
                    service=row.Service,
                    style=row.Style,
                    contest_type=row.Type,
                    framework=row.Framework,
                    target=ModelTarget.from_value(f"{key}{target_post}"),
                    features=row.Features,
                )
                + ".prediction.csv"
            )
            df = pd.read_csv(os.path.join(pred_dir_path, pred_filename))
            preds["pred." + key] = df.prediction

        lws_top_preds_df = pd.DataFrame(preds)

    crossovers = sum(lws_top_preds_df["pred.lws"] >= lws_top_preds_df["pred.top"])
    return crossovers / len(lws_top_preds_df)


def _transform(df: pd.DataFrame, pred_dir: str) -> pd.DataFrame:
    """
    Transform the evaluation results in the result file to a dataframe where each row
    is a pair of independant min cash and top score models or a single model that
    predicts for both min-cash and top score. Add cross over error rate and sort by
    combined MAE (mean)
    """
    assert not df.empty

    group_cols = ["Sport", "Service", "Type", "Style", "Framework", "Features"]
    out_cols = [
        "Sport",
        "Service",
        "Type",
        "Style",
        "Framework",
        "Features",
        "log",
        "orr",
        "combined_target",
        "pinball",
        "pinball.top",
        "pinball.lws",
        "MAE",
        "MAE.top",
        "MAE.lws",
        "R2",
        "R2.top",
        "R2.lws",
        "RMSE",
        "RMSE.top",
        "RMSE.lws",
    ]

    target_series = df["Target"].map(ModelTarget.from_value)

    # get the chained/combined models
    combined_mask = target_series.map(lambda t: t.is_combined)
    combined_df = df[combined_mask].copy()

    if combined_df.empty:
        combined_df = None
    else:
        combined_log = target_series[combined_mask].map(lambda t: t.is_log)
        combined_orr = target_series[combined_mask].map(lambda t: t.is_optrat_residual)
        combined_df = combined_df.assign(log=combined_log, orr=combined_orr).rename(
            columns={"Target": "combined_target"}
        )[out_cols]

    indiv = df[~combined_mask].copy()
    if indiv.empty:
        model_pair_df = combined_df
    else:
        # handle the dual model prediction pairs (i.e. seperate top and lws models that constitute the pair)
        indiv["log"] = target_series[~combined_mask].map(lambda t: t.is_log)
        indiv["orr"] = target_series[~combined_mask].map(lambda t: t.is_optrat_residual)
        indiv["is_top"] = target_series[~combined_mask].map(lambda t: t.is_top)

        idx_cols = group_cols + ["log", "orr"]
        top_df = indiv[indiv["is_top"]].set_index(idx_cols)
        lws_df = indiv[~indiv["is_top"]].set_index(idx_cols)

        already_combined = (
            set(map(tuple, combined_df[idx_cols].values.tolist()))
            if combined_df is not None
            else set()
        )

        rows = []
        for idx in top_df.index.intersection(lws_df.index):
            if idx in already_combined:
                continue
            top_model_info = top_df.loc[idx]
            lws_model_info = lws_df.loc[idx]
            sport, service, type_, style, framework, features, log, orr = idx

            r2_top, r2_lws = float(top_model_info["R2"]), float(lws_model_info["R2"])
            rmse_top, rmse_lws = float(top_model_info["RMSE"]), float(lws_model_info["RMSE"])
            mae_top, mae_lws = float(top_model_info["MAE"]), float(lws_model_info["MAE"])
            pb_top, pb_lws = float(top_model_info["pinball"]), float(lws_model_info["pinball"])

            rows.append(
                {
                    "Sport": sport,
                    "Service": service,
                    "Type": type_,
                    "Style": style,
                    "Framework": framework,
                    "Features": features,
                    "log": log,
                    "orr": orr,
                    "pinball": (pb_top + pb_lws) / 2,
                    "pinball.top": pb_top,
                    "pinball.lws": pb_lws,
                    "MAE": (mae_top + mae_lws) / 2,
                    "MAE.top": mae_top,
                    "MAE.lws": mae_lws,
                    "R2": (r2_top + r2_lws) / 2,
                    "R2.top": r2_top,
                    "R2.lws": r2_lws,
                    "RMSE": (rmse_top + rmse_lws) / 2,
                    "RMSE.top": rmse_top,
                    "RMSE.lws": rmse_lws,
                }
            )

        indiv_df = pd.DataFrame(rows, columns=out_cols) if rows else pd.DataFrame(columns=out_cols)
        model_pair_df = pd.concat([combined_df, indiv_df], ignore_index=True)

    tqdm.pandas(desc="adding xover-rate")
    crossover_rate = model_pair_df.progress_apply(_xo_rate, axis=1, args=(pred_dir,))
    model_pair_w_xover_df = model_pair_df.assign(crossover_rate=crossover_rate)
    sorted_df = model_pair_w_xover_df.sort_values(
        by=["Sport", "Service", "Type", "Style", "crossover_rate", "pinball", "pinball.lws"],
        ascending=True,
    )
    return sorted_df


_EVAL_RESULT_FILENAME_SEARCH_PATTERN = "all_eval_results-????????:??????.csv"


def _main(cmd_line_str=None):
    parser = ArgumentParser(
        description="Daily Fantasy winning score model evaluation results transformer. "
        "Load the evaluation result csv file generated during daily fantasy winning score "
        "model training. Produce a new tabular report organized by model pairs that can "
        "be used together, along with performance metrics for the pairs sorted in descending "
        "order by performance.",
        usage="If this is a directory as the argument then"
        f"'{_EVAL_RESULT_FILENAME_SEARCH_PATTERN}' will be searched for in the directory and "
        f"in the subdirectory '{WINSCORE_MODEL_RESULTS_SUBDIR}' if the subdirectory exists. "
        "The most recent matching file will be used. Test prediction result files "
        "(used to calculate crossover errors rate) will be searched for in the same directory "
        "that the file is located in.",
    )
    parser.add_argument(
        "eval_result_csv_filepath",
        help="Path to modeling evaluation results file or directory to search",
        type=pathlib.Path,
    )

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    arg_as_path = cast(pathlib.Path, args.eval_result_csv_filepath)
    if not arg_as_path.exists():
        parser.error(f"'{arg_as_path}' does not exist!")

    if arg_as_path.is_file():
        filepath = args.eval_result_csv_filepath
    else:
        assert arg_as_path.is_dir()
        paths_to_search = [
            arg_as_path / _EVAL_RESULT_FILENAME_SEARCH_PATTERN,
            arg_as_path / WINSCORE_MODEL_RESULTS_SUBDIR / _EVAL_RESULT_FILENAME_SEARCH_PATTERN,
        ]
        matched_files = chain(*[glob.glob(str(pattern)) for pattern in paths_to_search])
        sorted_matches = sorted(matched_files, reverse=True, key=os.path.basename)
        if len(sorted_matches) == 0:
            parser.error(f"No winscore evaluation results found. Searched in: {paths_to_search}")

        filepath = sorted_matches[0]
        if len(sorted_matches) > 1:
            print(f"""
{log.BOLD_RED}{len(sorted_matches)} matches found for search in searches:
{"\n".join(map(str, paths_to_search))}

The most recent will be transformed. Matches found:
*** {filepath} ***
{"\n".join(sorted_matches[1:])}{log.COLOR_RESET}
""")
        else:
            print(f"Transforming eval results at '{filepath}'")

    in_df = pd.read_csv(filepath)
    pred_dir = os.path.dirname(filepath)
    xformed_df = _transform(in_df, pred_dir)

    xformed_output_filename = os.path.basename(filepath).removesuffix(".csv")
    xformed_output_filepath = os.path.join(
        pred_dir, xformed_output_filename + ".xformed-lws-top.csv"
    )
    xformed_df.to_csv(xformed_output_filepath, index=False)
    print(tabulate(xformed_df, showindex=False, headers="keys"))
    print()
    print(f"transformed csv data written to '{xformed_output_filepath}'")


if __name__ == "__main__":
    _main()
