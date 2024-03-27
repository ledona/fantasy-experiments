import json
import logging
import os
import statistics
from typing import cast

import torch
from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    ContestStyle,
    FantasyException,
    dt_to_filename_str,
    log,
)
from fantasy_py.lineup import FantasyService
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loader import DeepLineupDataset
from .loss import DeepLineupLoss
from .model import DeepLineupModel, save

log.set_debug_log_level(__name__)
_LOGGER = log.get_logger(__name__)


DEFAULT_MODEL_FILENAME_FORMAT = "deep-lineup-model.{sport}.{service}.{style}.{datetime}.pkl"


class DeepTrainFailure(FantasyException):
    """raised if there is a failure during training"""


def _train_epoch(
    dataloader: DataLoader, model: DeepLineupModel, optimizer, deep_lineup_loss: DeepLineupLoss
):
    rewards = []
    model.train()

    for x, y in tqdm(dataloader, desc="batch", leave=False):
        # make predictions
        preds = model(x)

        # compute loss for the batch
        policy_gradients, reward = deep_lineup_loss(preds, y)

        if policy_gradients.isnan().any() or policy_gradients.isinf().any():
            _LOGGER.warning("nan or inf found in policy gradients")

        # back prop
        optimizer.zero_grad()
        # REINFORCE gradient update
        preds.backward(policy_gradients)

        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     for name, param in model.named_parameters():
        #         print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

        optimizer.step()
        rewards.append(reward)

    return rewards


def _eval_epoch(model, epoch, rewards):
    model.eval()
    _LOGGER.info(
        "batch rewards for epoch %i: mean=%.10f rewards=%s",
        epoch,
        statistics.mean(rewards),
        [round(reward, 5) for reward in rewards],
    )


def train(
    dataset_dir: str,
    train_epochs: int,
    batch_size: int,
    target_dir: str,
    model_filename: str | None = None,
    learning_rate=0.001,
):
    samples_meta_filepath = os.path.join(dataset_dir, "samples_meta.json")
    _LOGGER.info("Loading training samples from '%s'", samples_meta_filepath)
    with open(samples_meta_filepath, "r") as f_:
        samples_meta = json.load(f_)

    if model_filename is None:
        model_filename = DEFAULT_MODEL_FILENAME_FORMAT.format(
            sport=samples_meta["sport"],
            service=samples_meta["service"],
            style=samples_meta["style"],
            datetime=dt_to_filename_str(),
        )
    target_filepath = os.path.join(target_dir, model_filename)

    service_cls = cast(
        FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, samples_meta["service"])
    )
    constraints = service_cls.get_constraints(
        samples_meta["sport"], style=ContestStyle[samples_meta["style"]]
    )
    if constraints is None:
        raise DeepTrainFailure(
            f"Constraints not found for sport={samples_meta['sport']} "
            f"service={samples_meta['service']}"
        )

    dataset = DeepLineupDataset(samples_meta_filepath)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = DeepLineupModel(dataset.sample_df_len, len(dataset.input_cols))
    deep_lineup_loss = DeepLineupLoss(dataset, constraints)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(1, train_epochs + 1), desc="epoch"):
        rewards = _train_epoch(dataloader, model, optimizer, deep_lineup_loss)
        _eval_epoch(model, epoch, rewards)
        torch.save(model.state_dict(), target_filepath + ".epoch-" + str(epoch))

    _LOGGER.info("Training complete. Model written to '%s'", target_filepath)
    save(model, target_filepath)

    return (model,)
