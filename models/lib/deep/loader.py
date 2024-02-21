"""pytorch dataloader for deep learning lineup models"""

import json
import os
from typing import cast
from functools import cached_property

import pandas as pd
import torch
from torch.utils.data import Dataset


_PADDING_COST = 999999999
"""cost to use for padded inventory items"""

_DATASET_COLS_TO_DROP = ["in-lineup", "team_id"]


class DeepLineupDataset(Dataset):
    def __init__(self, samples_meta_filepath: str, padding=0.1):
        """
        padding: every dataframe returned will be padded to the length \
            of the longest dataframe plus this percent
        """
        self.data_dir = os.path.dirname(samples_meta_filepath)
        with open(samples_meta_filepath, "r") as f_:
            self.samples_meta = json.load(f_)

        max_len = max(info["items"] for info in self.samples_meta["samples"])
        self.sample_df_len = cast(int, max_len + int(max_len * padding))
        """the length of a sample dataframe plus padding"""

    def _get_sample_df(self, idx):
        sample_info = self.samples_meta["samples"][idx]
        filepath = os.path.join(
            self.data_dir,
            f"{sample_info['season']}-{sample_info['game_number']}-{sample_info['game_ids_hash']}.pq",
        )
        df = pd.read_parquet(filepath)
        return df

    @cached_property
    def target_cols(self):
        df = self._get_sample_df(0)
        target_cols = [
            col
            for col in df.columns
            if not col.startswith("pos:") and col not in ["fpts-predicted", "in-lineup"]
        ]
        return target_cols

    @cached_property
    def input_cols(self):
        df = self._get_sample_df(0)
        return [col for col in df if col not in _DATASET_COLS_TO_DROP]

    def __len__(self):
        return len(self.samples_meta["samples"])

    def __getitem__(self, idx):
        dataset_df = self._get_sample_df(idx)
        df = dataset_df.drop(columns=_DATASET_COLS_TO_DROP)
        if "player_id" in df:
            df = df.drop(columns="player_id")

        padding_df = pd.DataFrame((self.sample_df_len - len(df)) * [{"cost": _PADDING_COST}])
        df = pd.concat([df, padding_df]).fillna(0)
        tensor = torch.Tensor(df.values)
        target_df = dataset_df.query("`in-lineup`")[self.target_cols]
        return tensor, target_df.to_numpy()
