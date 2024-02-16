"""pytorch dataloader for deep learning lineup models"""

import json
import os
from typing import cast

import pandas as pd
import torch
from torch.utils.data import Dataset


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

    def __len__(self):
        return len(self.samples_meta["samples"])

    def __getitem__(self, idx):
        sample_info = self.samples_meta["samples"][idx]
        filepath = os.path.join(
            self.data_dir,
            f"{sample_info['season']}-{sample_info['game_number']}-{sample_info['game_ids_hash']}.pq",
        )
        df = pd.read_parquet(filepath).drop(columns="team_id")
        if "player_id" in df:
            df = df.drop(columns="player_id")

        padding_df = pd.DataFrame((self.sample_df_len - len(df)) * [{"cost": float("inf")}])
        df = pd.concat([df, padding_df]).fillna(0)
        tensor = torch.Tensor(df.values)
        return tensor, sample_info["top_score"]
