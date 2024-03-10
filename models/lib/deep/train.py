import json
import os
import statistics
from typing import cast
import logging

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


def _print_gradients(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter: {name}")
            print(param.grad)


def _train_epoch(
    dataloader: DataLoader, model: DeepLineupModel, optimizer, deep_lineup_loss: DeepLineupLoss
):
    losses = []
    model.train()

    for x, y in (batch_pbar := tqdm(dataloader, desc="batch", leave=False)):
        # make predictions
        preds = model(x)

        # compute loss
        loss = deep_lineup_loss(preds, y)

        # back prop
        optimizer.zero_grad()
        loss.backward()

        if _LOGGER.isEnabledFor(logging.DEBUG):
            _print_gradients(model)
        batch_pbar.set_postfix(loss=round(loss.item(), 5))

        optimizer.step()
        losses.append(loss.item())
        _LOGGER.debug("  batch loss: %s", round(loss.item(), 10))

    return losses


def _eval_epoch(model, epoch, losses):
    model.eval()
    _LOGGER.info("mean loss for epoch %i: %.5f", epoch, statistics.mean(losses))


def train(
    dataset_dir: str,
    train_epochs: int,
    batch_size: int,
    target_dir: str,
    model_filename: str | None = None,
    learning_rate=1e-3,
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
    deep_lineup_loss = DeepLineupLoss(dataset, constraints, model.to_inlineup)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(1, train_epochs + 1), desc="epoch"):
        losses = _train_epoch(dataloader, model, optimizer, deep_lineup_loss)
        _eval_epoch(model, epoch, losses)
        torch.save(model.state_dict(), target_filepath + ".epoch-" + str(epoch))

    _LOGGER.info("Training complete. Model written to '%s'", target_filepath)
    _LOGGER.info("Best lineup failure type and score: %s", deep_lineup_loss._best_lineup_found)
    save(model, target_filepath)

    return (model,)
