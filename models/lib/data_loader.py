"""pytorch dataloader for deep learning lineup models"""

import json
import os

import pandas as pd
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
        self.sample_df_len = max_len + int(max_len * padding)

    def __len__(self):
        return len(self.samples_meta["samples"])

    def __getitem__(self, idx):
        sample_info = self.samples_meta["samples"][idx]
        filepath = os.path.join(
            self.data_dir,
            f"{sample_info['season']}-{sample_info['game_number']}-{sample_info['game_ids_hash']}.pq",
        )
        df = pd.read_parquet(filepath)
        raise NotImplementedError('add padding')
        return df, sample_info["top_score"]
