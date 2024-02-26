import json
import os
import statistics
from typing import cast

import torch
from fantasy_py import FANTASY_SERVICE_DOMAIN, CLSRegistry, ContestStyle, FantasyException, log
from fantasy_py.lineup import FantasyService
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loader import DeepLineupDataset
from .loss import DeepLineupLoss
from .model import DeepLineupModel

_LOGGER = log.get_logger(__name__)


class DeepTrainFailure(FantasyException):
    """raised if there is a failure during training"""


def train(dataset_dir: str, train_epochs: int, batch_size: int):
    samples_meta_filepath = os.path.join(dataset_dir, "samples_meta.json")
    _LOGGER.info("Loading training samples from '%s'", samples_meta_filepath)
    with open(samples_meta_filepath, "r") as f_:
        samples_meta = json.load(f_)

    service_cls = cast(
        FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, samples_meta["service"])
    )
    constraints = service_cls.get_constraints(
        samples_meta["sport"], style=ContestStyle[samples_meta["style"]]
    )
    if constraints is None:
        raise DeepTrainFailure(
            f"Constraints not found for sport={samples_meta['sport']} service={samples_meta['service']}"
        )
    dataset = DeepLineupDataset(samples_meta_filepath)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    deep_lineup_loss = DeepLineupLoss(dataset.target_cols, constraints, dataset.cost_oom)
    model = DeepLineupModel(dataset.sample_df_len)

    # TODO: look inter optimizer options
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in tqdm(range(1, train_epochs + 1), desc="epoch"):
        losses = []

        for x, y in (batch_pbar := tqdm(dataloader, desc="batch", leave=False)):
            optimizer.zero_grad()

            # x has shape (batch_size, seq_len, input_size)
            preds = model(x)

            loss = deep_lineup_loss(preds, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            batch_pbar.set_postfix(loss=round(loss.item(), 2))

        _LOGGER.info("mean loss for epoch %i: %f", epoch, round(statistics.mean(losses), 2))

    return (
        model,
        samples_meta["sport"],
        samples_meta["service"],
        ContestStyle[samples_meta["style"]],
    )
