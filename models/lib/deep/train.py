import copy
import json
import os
import statistics
from datetime import datetime
from typing import Iterable, Literal, TypedDict, cast

import torch
from fantasy_py import (
    FANTASY_SERVICE_DOMAIN,
    CLSRegistry,
    ContestStyle,
    FantasyException,
    dt_to_filename_str,
    log,
)
from fantasy_py.lineup import DeepLineupDataset, DeepLineupModel, FantasyService
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import DeepLineupLoss

log.set_debug_log_level(__name__)
_LOGGER = log.get_logger(__name__)


DEFAULT_MODEL_FILENAME_FORMAT = "dlm.{sport}.{service}.{style}.{datetime}"


class DeepTrainFailure(FantasyException):
    """raised if there is a failure during training"""


def _train_epoch(
    dataloader: DataLoader, model: DeepLineupModel, optimizer, deep_lineup_loss: DeepLineupLoss
):
    rewards: list[float] = []
    model.train()

    for x, y in tqdm(dataloader, desc="batch", leave=False):
        # make predictions
        preds = model.forward(x)

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


_BestModel = tuple[int, float, None | DeepLineupModel]
"""best model thus far as a tuple of [epoch, score, model]"""


class CheckpointData(TypedDict):
    """model training checkpoint used to resume training"""

    best_model: _BestModel
    """the best model thus far"""
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


class CheckpointFileData(TypedDict):
    model: DeepLineupModel
    last_epoch: int
    addl_info: dict
    batch_size: int


def load_checkpoint(filepath: str):
    _LOGGER.info("Loading deep-lineup model from '%s'", filepath)
    return cast(CheckpointFileData, torch.load(filepath))


def save_checkpoint(
    model: DeepLineupModel,
    base_filepath: str,
    epoch: int,
    batch_size: int,
    new_best_score: bool,
    last_epoch: bool,
    **addl_info,
):
    """
    base_filepath: path + base filename (without extension) for model data files
    """
    checkpoint_filepath = base_filepath
    if new_best_score:
        checkpoint_filepath += "-best"
    if last_epoch:
        checkpoint_filepath += "-last"

    checkpoint_filepath += ".pt"
    _LOGGER.info("Saving checkpoint to '%s'", checkpoint_filepath)

    file_data: CheckpointFileData = {
        "model": model,
        "last_epoch": epoch,
        "addl_info": addl_info,
        "batch_size": batch_size,
    }
    torch.save(file_data, checkpoint_filepath)


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
        resume_dict = load_checkpoint(continue_from_checkpoint_filepath)
        starting_epoch = resume_dict["last_epoch"]

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
        best_model = checkpoint_data["best_model"]
        optimizer = torch.optim.Adam(model.nn_model.parameters(), lr=learning_rate)
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        _LOGGER.info(
            "Resuming from checkpoint at epoch=%i best-model-epoch=%i best-model-score=%f",
            starting_epoch,
            *best_model[:2],
        )
        _LOGGER.info("Model target filepath: '%s'", target_filepath)

    checkpoint_base_dict = {
        "dataset_dir": dataset_dir,
        "dataset_limit": dataset_limit,
    }
    dataset = DeepLineupDataset(samples_meta_filepath, limit=dataset_limit)

    if continue_from_checkpoint_filepath is None:
        assert batch_size is not None and hidden_size is not None
        best_model = cast(_BestModel, (-1, float("-inf"), None))
        epoch_scores = []
        starting_epoch = 0
        trained_on_dt = datetime.now()
        if model_filename is None:
            model_filename = DEFAULT_MODEL_FILENAME_FORMAT.format(
                sport=samples_meta["sport"],
                service=samples_meta["service"],
                style=samples_meta["style"],
                datetime=dt_to_filename_str(trained_on_dt),
            )
        target_filepath = os.path.join(target_dir, model_filename)

        model = DeepLineupModel(
            dataset.sample_df_len,
            dataset.input_cols,
            dataset.samples_meta["service"],
            dataset.samples_meta["sport"],
            hidden_size=hidden_size,
            dt_trained=trained_on_dt,
        )
        optimizer = torch.optim.Adam(model.nn_model.parameters(), lr=learning_rate)

    return (
        model,
        starting_epoch,
        best_model,
        epoch_scores,
        target_filepath,
        optimizer,
        checkpoint_base_dict,
        dataset,
        batch_size,
    )


def _validate_dataset_dirs(dir_path):
    """
    test that train and test directories and their samples_meta.json files exist

    returns dict[train|test, dirpath]
    """
    paths: dict[Literal["train", "test"], dict[Literal["dir", "meta"], str]] = {}
    for tt_type in ("train", "test"):
        dataset_dir = f"{dir_path}-{tt_type}"
        samples_meta_filepath = os.path.join(dataset_dir, "samples_meta.json")
        if not (os.path.isdir(dataset_dir) and os.path.isfile(samples_meta_filepath)):
            raise DeepTrainFailure(
                f"{tt_type}ing dataset dir '{dataset_dir}' is not a directory or "
                f"{tt_type}ing dataset metadata file '{samples_meta_filepath}' is not a file"
            )
        paths[tt_type] = {"dir": dataset_dir, "meta": samples_meta_filepath}
    return paths


def _calculate_performance(
    model: DeepLineupModel, deep_lineup_loss: DeepLineupLoss, test_meta_filepath: str
):
    _LOGGER.info("Calculating model performance against holdout")

    dataset = DeepLineupDataset(test_meta_filepath, sample_df_len=model.player_count)
    dataloader = DataLoader(dataset, batch_size=len(dataset))

    input_batch, target_batch = next(iter(dataloader))
    preds_batch = model.predict(input_batch)

    rewards: list[float] = []
    for preds, y in tqdm(zip(preds_batch, target_batch), desc="model-eval", total=len(dataset)):
        reward = deep_lineup_loss.calc_score(preds, y)
        rewards.append(float(reward))
    mean_reward = statistics.mean(rewards)
    return {
        "mean-reward": mean_reward,
        "samples": len(dataset),
        "seasons": dataset.samples_meta["seasons"],
    }


def train(
    dataset_dir_base: str,
    train_epochs: int,
    batch_size: int | None,
    target_dir: str,
    model_filename: str | None = None,
    learning_rate=0.001,
    hidden_size=128,
    checkpoint_epoch_interval: int = 5,
    continue_from_checkpoint_filepath: str | None = None,
    dataset_limit: int | None = None,
):
    """
    dataset_dir_base: there should be a training and testing folders named\
        {dataset_dir_base}-train and {dataset_dir_base}-test
    checkpoint_filepath: if not None, continue from the checkpoint at this filepath
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    _LOGGER.info("Device: %s", device)

    dataset_paths = _validate_dataset_dirs(dataset_dir_base)

    _LOGGER.info(
        "Training model: data-filepath-base='%s' learning-rate=%f "
        "hidden-size=%i batch-size=%i checkpoint-epoch-interval=%i",
        dataset_dir_base,
        learning_rate,
        hidden_size,
        batch_size,
        checkpoint_epoch_interval,
    )
    with open(dataset_paths["train"]["meta"], "r") as f_:
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
        best_model,
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
        dataset_dir_base,
        dataset_limit,
        hidden_size,
        dataset_paths["train"]["meta"],
        batch_size,
    )

    assert optimizer is not None and model.nn_model is not None

    dataloader = DataLoader(dataset, batch_size=batch_size)
    deep_lineup_loss = DeepLineupLoss(dataset, constraints)

    checkpoint_dirpath = target_filepath + "-checkpoints"
    if not os.path.isdir(checkpoint_dirpath):
        os.mkdir(checkpoint_dirpath)
        _LOGGER.info("Created model checkpoint directory '%s'", checkpoint_dirpath)

    for epoch_i in tqdm(
        range(train_epochs),
        total=train_epochs,
        # initial=starting_epoch,
        desc="epoch",
    ):
        if epoch_i < starting_epoch:
            continue
        rewards = _train_epoch(dataloader, model, optimizer, deep_lineup_loss)
        epoch_score = _eval_epoch(model, epoch_i, rewards)
        epoch_scores.append(epoch_score)
        if new_best_score := epoch_score > best_model[1]:
            _LOGGER.info(
                "New best model found! score=%f epoch=%i",
                epoch_score,
                epoch_i + 1,
            )
            best_model = (epoch_i + 1, epoch_score, copy.deepcopy(model))

        if (
            (epoch_i + 1) % checkpoint_epoch_interval == 0
            or new_best_score
            or epoch_i == train_epochs - 1
        ):
            checkpoint_filepath = os.path.join(checkpoint_dirpath, f"cp-epoch-{epoch_i + 1}")
            _LOGGER.info("Saving checkpoint for epoch %i", epoch_i + 1)
            save_checkpoint(
                model,
                checkpoint_filepath,
                epoch_i + 1,
                batch_size,
                new_best_score,
                epoch_i == train_epochs - 1,
                optimizer_state_dict=optimizer.state_dict(),
                epoch_scores=epoch_scores,
                best_model=best_model,
                target_filepath=target_filepath,
                **checkpoint_base_dict,
            )

    if best_model[2] is None:
        raise DeepTrainFailure("No best model found!")

    _LOGGER.info(
        "Training complete. Best model found in epoch=%i score=%f. mean-epoch-score=%f",
        *(*best_model[:2], statistics.mean(epoch_scores)),
    )

    performance = _calculate_performance(
        best_model[2], deep_lineup_loss, dataset_paths["test"]["meta"]
    )
    _LOGGER.info("Best model performance against hold-out: %s", performance["mean-reward"])
    best_model[2].save(
        target_filepath,
        best_model[0],
        batch_size,
        dataset,
        performance,
    )

    return model
