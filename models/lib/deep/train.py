import json
import os
import statistics
from typing import TypedDict, cast

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
from .model import DeepLineupModel, load, save

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
        epoch + 1,
        mean_reward,
        [round(reward, 5) for reward in rewards],
    )
    return mean_reward


_BestScore = tuple[int, float]
"""best score thus far as a tuple of [epoch, score]"""


class CheckpointData(TypedDict):
    best_score: _BestScore
    optimizer_state_dict: dict
    target_filepath: str
    epoch_scores: list[float]
    batch_size: int
    dataset_limit: int | None
    dataset_dir: str


def _checkpoint_cmdline_compare(name, cmdline_val, checkpoint_val):
    """
    make sure the cmdline value and checkpoint value match
    returns the value to use
    """
    if cmdline_val is None or checkpoint_val == cmdline_val:
        return checkpoint_val

    raise DeepTrainFailure(
        "Checkpoint '{name}' != commandline value. "
        "Remove '{name}' from commandline or make sure they match. "
        f"checkpoint-{name}={checkpoint_val} cmdline-{name}={cmdline_val}"
    )


def _setup_training(
    continue_from_checkpoint_filepath: None | str,
    target_dir: str,
    model_filename: str | None,
    learning_rate: float,
    samples_meta: dict,
    dataset_dir: None | str,
    dataset_limit: None | int,
    hidden_size: None | int,
    samples_meta_filepath,
    batch_size: None | int,
):
    if continue_from_checkpoint_filepath is not None:
        _LOGGER.info(
            "Resuming training from checkpoint file '%s'", continue_from_checkpoint_filepath
        )
        resume_dict = load(continue_from_checkpoint_filepath)
        starting_epoch = resume_dict["last_epoch"] + 1

        _checkpoint_cmdline_compare("hidden_size", hidden_size, resume_dict["model"]._hidden_size)
        batch_size = _checkpoint_cmdline_compare(
            "batch_size", batch_size, resume_dict["batch_size"]
        )
        model = resume_dict["model"]

        checkpoint_data = cast(CheckpointData, resume_dict["addl_info"])

        dataset_dir = _checkpoint_cmdline_compare(
            "dataset_dir", dataset_dir, checkpoint_data["dataset_dir"]
        )
        dataset_limit = _checkpoint_cmdline_compare(
            "dataset_limit", dataset_limit, checkpoint_data["dataset_limit"]
        )

        epoch_scores = checkpoint_data["epoch_scores"]

        target_filepath = checkpoint_data["target_filepath"]
        best_score = checkpoint_data["best_score"]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        _LOGGER.info(
            "Resuming from checkpoint at epoch=%i best-model-epoch=%i best-model-score=%f",
            starting_epoch,
            *best_score,
        )
    else:
        best_score: _BestScore = (-1, float("-inf"))
        epoch_scores = []
        starting_epoch = 0
        if model_filename is None:
            model_filename = DEFAULT_MODEL_FILENAME_FORMAT.format(
                sport=samples_meta["sport"],
                service=samples_meta["service"],
                style=samples_meta["style"],
                datetime=dt_to_filename_str(),
            )
        target_filepath = os.path.join(target_dir, model_filename)
        model = None
        optimizer = None

    _LOGGER.info("Model target filepath: '%s'", target_filepath)

    checkpoint_base_dict = {
        "dataset_dir": dataset_dir,
        "dataset_limit": dataset_limit,
    }

    dataset = DeepLineupDataset(samples_meta_filepath, limit=dataset_limit)

    return (
        model,
        starting_epoch,
        best_score,
        epoch_scores,
        target_filepath,
        optimizer,
        checkpoint_base_dict,
        dataset,
        batch_size,
    )


def train(
    dataset_dir: str,
    train_epochs: int,
    batch_size: int | None,
    target_dir: str,
    model_filename: str | None = None,
    learning_rate=0.001,
    hidden_size=128,
    checkpoint_epoch_interval: int | None = 5,
    continue_from_checkpoint_filepath: str | None = None,
    dataset_limit: int | None = None,
):
    """
    checkpoint_filepath: if not None, continue from the checkpoint at this filepath
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    _LOGGER.info("Device: %s", device)

    samples_meta_filepath = os.path.join(dataset_dir, "samples_meta.json")
    _LOGGER.info(
        "Training model: data-filepath='%s' learning-rate=%f "
        "hidden-size=%i batch-size=%i checkpoint-epoch-interval=%i",
        samples_meta_filepath,
        learning_rate,
        hidden_size,
        batch_size,
        checkpoint_epoch_interval,
    )
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
            f"Constraints not found for sport={samples_meta['sport']} "
            f"service={samples_meta['service']}"
        )

    (
        model,
        starting_epoch,
        best_score,
        epoch_scores,
        target_filepath,
        optimizer,
        checkpoint_base_dict,
        dataset,
        batch_size,
    ) = _setup_training(
        continue_from_checkpoint_filepath,
        target_dir,
        model_filename,
        learning_rate,
        samples_meta,
        dataset_dir,
        dataset_limit,
        hidden_size,
        samples_meta_filepath,
        batch_size,
    )

    assert (model is None) == (optimizer is None)
    if model is None:
        # should be a new model, not continuing from a checkpoint
        assert optimizer is None and continue_from_checkpoint_filepath is None
        model = DeepLineupModel(
            dataset.sample_df_len, len(dataset.input_cols), hidden_size=hidden_size
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    assert optimizer is not None

    model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    deep_lineup_loss = DeepLineupLoss(dataset, constraints)

    for epoch_i in tqdm(
        range(train_epochs),
        total=train_epochs,
        initial=starting_epoch,
        desc="epoch",
    ):
        rewards = _train_epoch(dataloader, model, optimizer, deep_lineup_loss)
        epoch_score = _eval_epoch(model, epoch_i, rewards)
        epoch_scores.append(epoch_score)
        if new_best_score := epoch_score > best_score[1]:
            _LOGGER.info(
                "New best model found! score=%f epoch=%i",
                epoch_score,
                epoch_i + 1,
            )
            best_score = (epoch_i + 1, epoch_score)
            save(
                target_filepath + ".pt",
                model,
                epoch_i + 1,
                batch_size,
                model_description="best-score",
            )

        if (
            checkpoint_epoch_interval is not None
            and epoch_i > 0
            and ((epoch_i + 1) % checkpoint_epoch_interval == 0 or new_best_score)
        ):
            checkpoint_filepath = target_filepath + f".checkpoint.epoch-{epoch_i + 1}.pt"
            _LOGGER.info("Saving checkpoint for epoch %i", epoch_i + 1)
            save(
                checkpoint_filepath,
                model,
                epoch_i + 1,
                batch_size,
                model_description=f"checkpoint-epoch-{epoch_i + 1}",
                optimizer_state_dict=optimizer.state_dict(),
                epoch_scores=epoch_scores,
                best_score=best_score,
                target_filepath=target_filepath,
                **checkpoint_base_dict,
            )

    _LOGGER.info(
        "Training complete. Best model found in epoch=%i score=%f. mean-epoch-score=%f",
        *(*best_score, statistics.mean(epoch_scores)),
    )

    return model
