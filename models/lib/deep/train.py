import json
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


DEFAULT_MODEL_FILENAME_FORMAT = "deep-lineup-model.{sport}.{service}.{style}.{datetime}"


class DeepTrainFailure(FantasyException):
    """raised if there is a failure during training"""


def _train_epoch(
    dataloader: DataLoader, model: DeepLineupModel, optimizer, deep_lineup_loss: DeepLineupLoss
):
    rewards: list[float] = []
    model.train()

    for x, y in tqdm(dataloader, desc="batch", leave=False):
        # make predictions
        preds = model(x)

        # compute loss for the batch
        loss, reward = cast(tuple[torch.Tensor, float], deep_lineup_loss(preds, y))

        if loss.isnan().any() or loss.isinf().any():
            _LOGGER.warning("nan or inf found in policy gradients")

        # back prop
        optimizer.zero_grad()
        deep_lineup_loss.backwards_(preds, loss)

        # if _LOGGER.isEnabledFor(logging.DEBUG):
        #     for name, param in model.named_parameters():
        #         print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]}")

        optimizer.step()
        rewards.append(reward)

    return rewards


def _eval_epoch(model, epoch, rewards: list[float]):
    model.eval()
    mean_reward = statistics.mean(rewards)
    _LOGGER.info(
        "batch rewards for epoch %i: mean=%.10f rewards=%s",
        epoch,
        mean_reward,
        [round(reward, 5) for reward in rewards],
    )
    return mean_reward


def train(
    dataset_dir: str,
    train_epochs: int,
    batch_size: int,
    target_dir: str,
    model_filename: str | None = None,
    learning_rate=0.001,
    hidden_size=128,
    checkpoint_epoch_interval: int | None = 5,
    continue_from_checkpoint_filepath: str | None = None,
):
    """
    checkpoint_filepath: if not None, continue from the checkpoint at this filepath
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples_meta_filepath = os.path.join(dataset_dir, "samples_meta.json")
    _LOGGER.info(
        "Training model: device=%s checkpoint-restart-file='%s' data-filepath='%s' learning-rate=%f "
        "hidden-size=%i batch-size=%i checkpoint-epoch-interval=%i",
        device,
        continue_from_checkpoint_filepath,
        samples_meta_filepath,
        learning_rate,
        hidden_size,
        batch_size,
        checkpoint_epoch_interval,
    )
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
    _LOGGER.info("Model target filepath: '%s'", target_filepath)

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
    model = DeepLineupModel(
        dataset.sample_df_len, len(dataset.input_cols), hidden_size=hidden_size
    ).to(device)
    deep_lineup_loss = DeepLineupLoss(dataset, constraints)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if continue_from_checkpoint_filepath is not None:
        raise NotImplementedError()
    else:
        best_score = (-1, float("-inf"))
        epoch_scores: list[float] = []
        starting_epoch = 1

    for epoch in tqdm(range(starting_epoch, train_epochs + 1), desc="epoch"):
        rewards = _train_epoch(dataloader, model, optimizer, deep_lineup_loss)
        epoch_score = _eval_epoch(model, epoch, rewards)
        epoch_scores.append(epoch_score)
        if epoch_score > best_score[1]:
            _LOGGER.info(
                "New best model found! old[epoch=%i score=%f] new[epoch=%i score=%f]",
                *(*best_score, epoch, epoch_score),
            )
            best_score = (epoch, epoch_score)
            save(target_filepath + ".pt", model, epoch, rewards)

        if checkpoint_epoch_interval is not None and epoch % checkpoint_epoch_interval == 0:
            _LOGGER.info("Saving checkpoint at epoch %i", epoch)
            save(
                target_filepath + f".checkpoint.epoch-{epoch}.pt",
                model,
                epoch,
                rewards,
                optimizer=optimizer.state_dict(),
                epoch_scores=epoch_scores,
                best_score=best_score,
            )

    _LOGGER.info(
        "Training complete. Best model found in epoch=%i score=%f. mean-epoch-score=%f",
        *(*best_score, statistics.mean(epoch_scores)),
    )

    return model
