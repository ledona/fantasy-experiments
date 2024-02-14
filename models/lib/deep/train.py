import json
import os
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
    lineup_constraints = constraints.lineup_constraints
    if isinstance(lineup_constraints, dict):
        lineup_slot_count = sum(lineup_constraints.values())
    elif isinstance(lineup_constraints, list):
        lineup_slot_count = sum(lineup_constraints)
    else:
        lineup_slot_count = sum(constraint.max_count for constraint in lineup_constraints)

    dataset = DeepLineupDataset(samples_meta_filepath)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss = DeepLineupLoss()
    model = DeepLineupModel(lineup_slot_count)

    # TODO: look inter optimizer options
    optimizer = torch.optim.Adam(model.parameters())

    for _ in tqdm(range(train_epochs, 1, train_epochs + 1), desc="deep-training-epoch"):
        for x, y in dataloader:
            optimizer.zero_grad()

            # x has shape (batch_size, seq_len, input_size)
            preds = model(x)

            loss = loss(preds, y)
            loss.backward()
            optimizer.step()

    return model