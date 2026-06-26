"""
Load model eval result csv file, reorganize the data by pairing
complementary (pairable) top and lws models along with the pair's
performance metrics, save the model top+lws model pair evaluation results
in a way that is easier to review than the original eval results AND
if such that it can easily by used/reused to identify the model pairs
that should be used when backtesting"""

import os
import shlex
from argparse import ArgumentParser

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from .model import model_filenamer


def _xo_rate(row: pd.Series, pred_dir_path: str):
    """pandas apply function for calculating crossover rate"""
    lws_top_preds_df: pd.DataFrame
    target_post = "_log" if row.log else ""

    if "regchain" in row.Framework:
        pred_filename = (
            model_filenamer(
                sport=row.Sport,
                service=row.Service,
                style=row.Style,
                contest_type=row.Type,
                framework=row.Framework,
                target="top+lws" + target_post,
                features=row.Features,
            )
            + ".prediction.csv"
        )
        lws_top_preds_df = pd.read_csv(os.path.join(pred_dir_path, pred_filename))
    else:
        preds = {}
        for target in ["top", "lws"]:
            pred_filename = (
                model_filenamer(
                    sport=row.Sport,
                    service=row.Service,
                    style=row.Style,
                    contest_type=row.Type,
                    framework=row.Framework,
                    target=target + target_post,
                    features=row.Features,
                )
                + ".prediction.csv"
            )
            df = pd.read_csv(os.path.join(pred_dir_path, pred_filename))
            preds["pred." + target] = df.prediction

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

    group_cols = ["Sport", "Service", "Type", "Style", "Framework", "Features"]
    out_cols = [
        "Sport",
        "Service",
        "Type",
        "Style",
        "Framework",
        "Features",
        "log",
        "R2",
        "R2.top",
        "R2.lws",
        "RMSE",
        "RMSE.top",
        "RMSE.lws",
        "MAE",
        "MAE.top",
        "MAE.lws",
    ]

    paired_mask = df["Target"].str.contains("+", regex=False)

    paired = df[paired_mask].copy()
    paired["log"] = paired["Target"].str.endswith("_log")
    paired_out = paired[out_cols]

    indiv = df[~paired_mask].copy()
    indiv["log"] = indiv["Target"].str.endswith("_log")
    indiv["is_top"] = indiv["Target"].str.startswith("top")

    idx_cols = group_cols + ["log"]
    top_df = indiv[indiv["is_top"]].set_index(idx_cols)
    lws_df = indiv[~indiv["is_top"]].set_index(idx_cols)

    already_paired = set(map(tuple, paired[idx_cols].values.tolist()))

    rows = []
    for idx in top_df.index.intersection(lws_df.index):
        if idx in already_paired:
            continue
        top_model_info = top_df.loc[idx]
        lws_model_info = lws_df.loc[idx]
        sport, service, type_, style, framework, features, log = idx
        r2_top, r2_lws = float(top_model_info["R2"]), float(lws_model_info["R2"])
        rmse_top, rmse_lws = float(top_model_info["RMSE"]), float(lws_model_info["RMSE"])
        mae_top, mae_lws = float(top_model_info["MAE"]), float(lws_model_info["MAE"])
        rows.append(
            {
                "Sport": sport,
                "Service": service,
                "Type": type_,
                "Style": style,
                "Framework": framework,
                "Features": features,
                "log": log,
                "R2": (r2_top + r2_lws) / 2,
                "R2.top": r2_top,
                "R2.lws": r2_lws,
                "RMSE": (rmse_top + rmse_lws) / 2,
                "RMSE.top": rmse_top,
                "RMSE.lws": rmse_lws,
                "MAE": (mae_top + mae_lws) / 2,
                "MAE.top": mae_top,
                "MAE.lws": mae_lws,
            }
        )

    indiv_out = pd.DataFrame(rows, columns=out_cols) if rows else pd.DataFrame(columns=out_cols)
    model_pair_df = pd.concat([paired_out, indiv_out], ignore_index=True)

    tqdm.pandas(desc="adding xover-rate")
    crossover_rate = model_pair_df.progress_apply(_xo_rate, axis=1, args=(pred_dir,))
    model_pair_w_xover_df = model_pair_df.assign(crossover_rate=crossover_rate)
    sorted_df = model_pair_w_xover_df.sort_values(
        by=["Sport", "Service", "Type", "Style", "crossover_rate", "MAE", "MAE.lws"],
        ascending=[True, True, True, True, True, True, True],
    )
    return sorted_df


def _main(cmd_line_str=None):
    parser = ArgumentParser(
        description="Daily Fantasy winning score model evaluation results transformer. "
        "Load the evaluation result csv file generated during daily fantasy winning score "
        "model training. Produce a new tabular report organized by model pairs that can "
        "be used together, along with performance metrics for the pairs sorted in descending "
        "order by performance."
    )
    parser.add_argument(
        "eval_result_csv_filepath",
        help="Path to modeling evaluation results file. "
        "Test prediction result files (used to calculate crossover errors rate) will "
        "be searched for in the same directory.",
    )

    arg_strings = shlex.split(cmd_line_str) if cmd_line_str is not None else None
    args = parser.parse_args(arg_strings)

    in_df = pd.read_csv(args.eval_result_csv_filepath)
    pred_dir = os.path.dirname(args.eval_result_csv_filepath)
    xformed_df = _transform(in_df, pred_dir)

    xformed_output_filename = os.path.basename(args.eval_result_csv_filepath)
    if xformed_output_filename.endswith(".csv"):
        xformed_output_filename = xformed_output_filename[:-4]
    xformed_output_filepath = os.path.join(
        pred_dir, xformed_output_filename + ".xformed-lws-top.csv"
    )
    xformed_df.to_csv(xformed_output_filepath, index=False)
    print(tabulate(xformed_df, showindex=False, headers="keys"))
    print()
    print(f"transformed csv data written to '{xformed_output_filepath}'")


if __name__ == "__main__":
    _main()
