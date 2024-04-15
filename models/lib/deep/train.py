import copy
import json
import os
import statistics
from datetime import datetime
from typing import Literal, TypedDict, cast

import torch
from fantasy_py import FANTASY_SERVICE_DOMAIN, CLSRegistry, ContestStyle, FantasyException, log
from fantasy_py.lineup import DeepLineupDataset, DeepLineupModel, FantasyService
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import DeepLineupLoss

log.set_debug_log_level(__name__)
_LOGGER = log.get_logger(__name__)


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

    model.epochs_trained += 1
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


_BestModel = tuple[float, None | DeepLineupModel]
"""best model thus far as a tuple of [score, model]"""


class CheckpointBaseData(TypedDict):
    dataset_limit: int | None
    dataset_dir: str


class CheckpointData(CheckpointBaseData):
    """model training checkpoint used to resume training"""

    model: DeepLineupModel
    best_model: _BestModel
    """the best model thus far"""
    optimizer_state_dict: dict
    target_filepath: str
    epoch_scores: list[float]


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


def _save_checkpoint(
    base_filepath: str,
    new_best_score: bool,
    last_epoch: bool,
    early_stop: bool,
    checkpoint_data: CheckpointData,
):
    """
    base_filepath: path + base filename (without extension) for model data files
    last_epoch: if true, checkpoint is for last training epoch
    """
    checkpoint_filepath = base_filepath
    if new_best_score:
        checkpoint_filepath += "-best"
    if last_epoch:
        checkpoint_filepath += "-last"
    if early_stop:
        checkpoint_filepath += "-early-stop"

    checkpoint_filepath += ".pt"
    _LOGGER.info("Saving checkpoint to '%s'", checkpoint_filepath)

    torch.save(checkpoint_data, checkpoint_filepath)


def _setup_training(
    continue_from_checkpoint_filepath: None | str,
    target_dir: str,
    learning_rate: float,
    samples_meta: dict,
    dataset_dir: None | str,
    dataset_limit: None | int,
    hidden_size: None | int,
    samples_meta_filepath,
    batch_size: None | int,
    max_epochs: None | int,
    overwrite: bool,
):
    if continue_from_checkpoint_filepath is not None:
        _LOGGER.info(
            "Resuming training from checkpoint file '%s'", continue_from_checkpoint_filepath
        )
        checkpoint_data = cast(CheckpointData, torch.load(continue_from_checkpoint_filepath))
        starting_epoch = checkpoint_data["model"].epochs_trained

        _checkpoint_cmdline_compare(
            "hidden_size", hidden_size, checkpoint_data["model"].hidden_size
        )
        model = checkpoint_data["model"]
        _checkpoint_cmdline_compare("batch_size", batch_size, model.batch_size)

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
        assert best_model[1] is not None
        _LOGGER.info(
            "Resuming from checkpoint at epoch=%i best-model-epoch=%i best-model-score=%f",
            starting_epoch,
            best_model[1].epochs_trained,
            best_model[0],
        )
        _LOGGER.info("Model target filepath: '%s'", target_filepath)

    checkpoint_base_data = cast(
        CheckpointBaseData,
        {
            "dataset_dir": dataset_dir,
            "dataset_limit": dataset_limit,
        },
    )
    dataset = DeepLineupDataset(samples_meta_filepath, limit=dataset_limit)

    if continue_from_checkpoint_filepath is None:
        assert batch_size is not None and hidden_size is not None and max_epochs is not None
        best_model = cast(_BestModel, (float("-inf"), None))
        epoch_scores = []
        starting_epoch = 0
        trained_on_dt = datetime.now()
        model_filename = DeepLineupModel.default_name(
            samples_meta["sport"],
            samples_meta["service"],
            samples_meta["style"],
            dt_trained=trained_on_dt,
        )
        target_filepath = os.path.join(target_dir, model_filename)

        model = DeepLineupModel(
            dataset.input_cols,
            dataset.samples_meta["sport"],
            dataset.samples_meta["service"],
            samples_meta["style"],
            dataset.sample_df_len,
            learning_rate,
            batch_size,
            hidden_size,
            max_epochs,
            trained_on_dt,
            model_dependencies=dataset.samples_meta["models"],
        )
        optimizer = torch.optim.Adam(model.nn_model.parameters(), lr=learning_rate)

    assert isinstance(target_filepath, str)
    if not overwrite and os.path.exists(target_filepath + ".model"):
        raise DeepTrainFailure(
            f"Model file '{target_filepath}.model' already exists. Use --overwrite"
        )

    return (
        model,
        starting_epoch,
        best_model,
        epoch_scores,
        target_filepath,
        optimizer,
        checkpoint_base_data,
        dataset,
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
    _LOGGER.info("Calculating model performance against holdout at '%s'", test_meta_filepath)

    dataset = DeepLineupDataset(test_meta_filepath, sample_df_len=model.player_count)
    dataloader = DataLoader(dataset, batch_size=len(dataset))

    input_batch, target_batch = next(iter(dataloader))
    preds_batch = model.predict(input_batch)

    rewards: list[float] = []
    for preds, y in tqdm(zip(preds_batch, target_batch), desc="model-eval", total=len(dataset)):
        reward = deep_lineup_loss.calc_score(preds, y)
        rewards.append(float(reward))
    mean_reward = statistics.mean(rewards)
    model.performance = {
        "mean-reward": mean_reward,
        "samples": len(dataset),
        "seasons": dataset.samples_meta["seasons"],
    }
    return model.performance


def train(
    dataset_dir_base: str,
    train_epochs: int,
    batch_size: int | None,
    target_dir: str,
    learning_rate=0.001,
    hidden_size=128,
    checkpoint_epoch_interval: int = 5,
    continue_from_checkpoint_filepath: str | None = None,
    dataset_limit: int | None = None,
    early_stopping_patience: int | None = None,
    overwrite: bool = False,
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
        "hidden-size=%i batch-size=%i checkpoint-epoch-interval=%i "
        "early-stopping-patience=%s",
        dataset_dir_base,
        learning_rate,
        hidden_size,
        batch_size,
        checkpoint_epoch_interval,
        early_stopping_patience,
    )
    with open(dataset_paths["train"]["meta"], "r") as f_:
        samples_meta = json.load(f_)

    service_cls = cast(
        FantasyService, CLSRegistry.get_class(FANTASY_SERVICE_DOMAIN, samples_meta["service"])
    )
    constraints = service_cls.get_constraints(
        samples_meta["sport"], style=ContestStyle(samples_meta["style"])
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
    ) = _setup_training(
        continue_from_checkpoint_filepath,
        target_dir,
        learning_rate,
        samples_meta,
        dataset_dir_base,
        dataset_limit,
        hidden_size,
        dataset_paths["train"]["meta"],
        batch_size,
        train_epochs,
        overwrite,
    )

    assert optimizer is not None and model.nn_model is not None

    # TODO: play with this to optimize for performance
    dataloader = DataLoader(
        dataset,
        batch_size=model.batch_size,
        shuffle=False,
        # num_workers=4
        # pin_memory=True,
        # pin_memory_device=device,
        # generator=torch.Generator(device=device)
    )
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
        if new_best_score := epoch_score > best_model[0]:
            _LOGGER.info(
                "*** New best model found! score=%f epoch=%i ***",
                epoch_score,
                epoch_i + 1,
            )
            best_model = (epoch_score, copy.deepcopy(model))

        if stop_early := (
            early_stopping_patience is not None
            and not new_best_score
            and best_model[1] is not None
            and epoch_i + 1 - best_model[1].epochs_trained >= early_stopping_patience
        ):
            _LOGGER.info(
                "Early stopping triggered at epoch=%i. "
                "Last new best model was %i epochs ago at epoch=%i",
                epoch_i + 1,
                early_stopping_patience,
                best_model[1].epochs_trained,
            )

        if (
            (epoch_i + 1) % checkpoint_epoch_interval == 0
            or new_best_score
            or stop_early
            or epoch_i == train_epochs - 1
        ):
            checkpoint_filepath = os.path.join(checkpoint_dirpath, f"cp-epoch-{epoch_i + 1}")
            _LOGGER.info("Saving checkpoint for epoch %i", epoch_i + 1)
            checkpoint_dict: CheckpointData = {
                "model": model,
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch_scores": epoch_scores,
                "best_model": best_model,
                "target_filepath": target_filepath,
                **checkpoint_base_dict,
            }
            _save_checkpoint(
                checkpoint_filepath,
                new_best_score,
                epoch_i == train_epochs - 1,
                stop_early,
                checkpoint_dict,
            )

        if stop_early:
            break

    if best_model[1] is None:
        raise DeepTrainFailure("No best model found!")

    _LOGGER.info(
        "Training complete. Best model found at epoch=%i score=%f. mean-epoch-score=%f",
        best_model[1].epochs_trained,
        best_model[0],
        statistics.mean(epoch_scores),
    )

    performance = _calculate_performance(
        best_model[1], deep_lineup_loss, dataset_paths["test"]["meta"]
    )
    _LOGGER.info("Best model performance against hold-out: %s", performance["mean-reward"])
    best_model[1].dump(target_filepath, overwrite=overwrite)

    return model
