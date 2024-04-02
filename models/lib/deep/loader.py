"""pytorch dataloader for deep learning lineup models"""

import json
import os
from functools import cached_property
from typing import cast

import pandas as pd
import torch
from fantasy_py import DataNotAvailableException, UnexpectedValueError
from torch.utils.data import Dataset

_PADDING_COST = 999999999
"""cost to use for padded inventory items"""

_DATASET_COLS_TO_DROP = ["fpts-historic", "in-lineup"]
"""
columns to drop from the dataset before creating an input tensor
'player_id' will also be dropped if present
"""


class DeepLineupDataset(Dataset):
    def __init__(
        self,
        samples_meta_filepath: str,
        padding=0.1,
        sample_df_len: None | int = None,
        limit: None | int = None,
    ):
        """
        sample_df_len: the length of samples returned by the dataset, \
            if None the this will be calculated using the padding parameter.\
            If not None then the padding parameter will be ignored.
        padding: every dataframe returned will be padded to the length \
            of the longest dataframe plus this percent.
        limit: Limit the dataset to this number of samples
        """
        self.limit = limit
        self.data_dir = os.path.dirname(samples_meta_filepath)
        with open(samples_meta_filepath, "r") as f_:
            self.samples_meta = json.load(f_)

        if limit is not None and limit > (n := len(self.samples_meta["samples"])):
            raise DataNotAvailableException(
                f"Limit of {limit} is greater than the available data (n={n})"
            )

        max_len = max(info["items"] for info in self.samples_meta["samples"])
        self.sample_df_len = sample_df_len or cast(int, max_len + int(max_len * padding))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_sample_df(self, idx, _test_cols=True):
        sample_info = self.samples_meta["samples"][idx]
        filepath = os.path.join(
            self.data_dir,
            f"{sample_info['season']}-{sample_info['game_number']}-{sample_info['game_ids_hash']}.pq",
        )
        df = pd.read_parquet(filepath)
        assert not _test_cols or sorted(df.columns) == sorted(self.target_cols), (
            f"dataset sample {idx=} columns do not match columns from sample 0: "
            f"{sorted(df.columns)} != {sorted(self.target_cols)}"
        )
        return df

    @cached_property
    def target_cols(self):
        df = self._get_sample_df(0, False)
        return df.columns

    @cached_property
    def input_cols(self):
        df = self._get_sample_df(0)
        ignore_cols = (
            _DATASET_COLS_TO_DROP
            if "player_id" not in df
            else ["player_id", *_DATASET_COLS_TO_DROP]
        )
        return [str(col) for col in df if col not in ignore_cols]

    def __len__(self):
        return self.limit or len(self.samples_meta["samples"])

    def _get_padded_df(self, df: pd.DataFrame):
        """pad the dataframe to the required number of rows"""
        if len(df) > self.sample_df_len:
            raise UnexpectedValueError(
                "The dataframe is already greater than the target number of rows"
            )
        padding_df = pd.DataFrame((self.sample_df_len - len(df)) * [{"cost": _PADDING_COST}])
        return pd.concat([df, padding_df]).fillna(0)

    def __getitem__(self, idx):
        """
        returns - tuple of (input, target) where input is the input tensor for model training \
            and target is a numpy array describing all information\
            about the slate including the target/optimal lineup
        """
        target_df = self._get_sample_df(idx).fillna(0)
        padded_target_df = self._get_padded_df(target_df)
        input_df = padded_target_df[self.input_cols]
        input_tensor = torch.Tensor(input_df.values).to(self.device)
        assert len(input_tensor) == self.sample_df_len

        bool_fixed_target_df = padded_target_df.replace(
            {"in-lineup": {False: 0.0, True: 1.0}}
        ).astype(float)
        target_tensor = torch.Tensor(bool_fixed_target_df.values).to(self.device)
        assert len(target_tensor) == self.sample_df_len

        return input_tensor, target_tensor
