"""pytorch dataloader for deep learning lineup models"""

import json
import math
import os
from functools import cached_property
from typing import cast

import pandas as pd
import torch
from fantasy_py import FantasyException
from torch.utils.data import Dataset

_PADDING_COST = 999999999
"""cost to use for padded inventory items"""

_DATASET_COLS_TO_DROP = ["fpts-historic", "in-lineup"]
"""
columns to drop from the dataset before creating an input tensor
'player_id' will also be dropped if present
"""


class InsufficientData(FantasyException):
    """raised if the requested dataset limit is greater than the available data"""


class DeepLineupDataset(Dataset):
    def __init__(self, samples_meta_filepath: str, padding=0.1, limit: None | int = None):
        """
        padding: every dataframe returned will be padded to the length \
            of the longest dataframe plus this percent
        limit: Limit the dataset to this number of samples
        """
        self.limit = limit
        self.data_dir = os.path.dirname(samples_meta_filepath)
        with open(samples_meta_filepath, "r") as f_:
            self.samples_meta = json.load(f_)

        if limit is not None and limit > (n := len(self.samples_meta["samples"])):
            raise InsufficientData(f"Limit of {limit} is greater than the available data (n={n})")

        max_len = max(info["items"] for info in self.samples_meta["samples"])
        self.sample_df_len = cast(int, max_len + int(max_len * padding))
        """the length of a sample dataframe plus padding"""

    def _get_sample_df(self, idx, _test_cols=True):
        sample_info = self.samples_meta["samples"][idx]
        filepath = os.path.join(
            self.data_dir,
            f"{sample_info['season']}-{sample_info['game_number']}-{sample_info['game_ids_hash']}.pq",
        )
        df = pd.read_parquet(filepath)
        assert not _test_cols or set(df.columns) == set(self.target_cols)
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

    @cached_property
    def cost_oom(self):
        """return an order of magnitude estimate of the cost of the dataset"""
        cost_mean = self._get_sample_df(0)["cost"].mean()
        return int(math.log(cost_mean, 10))

    def __len__(self):
        return self.limit or len(self.samples_meta["samples"])

    def __getitem__(self, idx):
        """
        returns - tuple of (input, target) where input is the input tensor for model training \
            and target is a numpy array describing all information\
            about the slate including the target/optimal lineup
        """
        target_df = self._get_sample_df(idx)
        padding_df = pd.DataFrame((self.sample_df_len - len(target_df)) * [{"cost": _PADDING_COST}])
        target_df = pd.concat([target_df, padding_df]).fillna(0)

        input_df = target_df[self.input_cols]
        target_df.replace(False, 0.0, inplace=True)
        target_df.replace(True, 1.0, inplace=True)

        tensor = torch.Tensor(input_df.values)

        assert len(tensor) == self.sample_df_len
        assert len(target_df) == self.sample_df_len
        return tensor, target_df.to_numpy()
