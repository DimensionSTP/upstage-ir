import os

import pandas as pd

from hydra.utils import instantiate
from omegaconf import DictConfig

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

from ..utils.setup import SetUp
from ..tuners.huggingface_tuner import HuggingFaceTuner


def train(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    try:
        if isinstance(config.resumed_step, int):
            if config.resumed_step == 0:
                trainer.fit(
                    model=architecture,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                )
            elif config.resumed_step > 0:
                trainer.fit(
                    model=architecture,
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader,
                    ckpt_path=f"{config.callbacks.model_checkpoint.dirpath}/last.ckpt",
                )
            else:
                raise ValueError(
                    f"Invalid resumed_step argument: {config.resumed_step}"
                )
        else:
            raise TypeError(f"Invalid resumed_step argument: {config.resumed_step}")
        logger.experiment.alert(
            title="Training Complete",
            text="Training process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Training Error",
            text="An error occurred during training",
            level="ERROR",
        )
        raise e

    if (
        config.strategy == "deepspeed_stage_3"
        or config.strategy == "deepspeed_stage_3_offload"
    ):
        for epoch in range(config.epoch):
            ckpt_path = f"{config.callbacks.model_checkpoint.dirpath}/epoch{epoch}.ckpt"
            if os.path.exists(ckpt_path) and os.path.isdir(ckpt_path):
                convert_zero_checkpoint_to_fp32_state_dict(
                    ckpt_path,
                    f"{ckpt_path}/model.pt",
                )


def test(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    test_loader = setup.get_test_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    if (
        config.strategy == "deepspeed_stage_3"
        or config.strategy == "deepspeed_stage_3_offload"
    ):
        trainer: Trainer = instantiate(
            config.trainer,
            strategy="ddp",
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )
    else:
        trainer: Trainer = instantiate(
            config.trainer,
            callbacks=callbacks,
            logger=logger,
            _convert_="partial",
        )

    try:
        if (
            config.strategy == "deepspeed_stage_3"
            or config.strategy == "deepspeed_stage_3_offload"
        ):
            trainer.test(
                model=architecture,
                dataloaders=test_loader,
                ckpt_path=f"{config.ckpt_path}/model.pt",
            )
        else:
            trainer.test(
                model=architecture,
                dataloaders=test_loader,
                ckpt_path=config.ckpt_path,
            )
        logger.experiment.alert(
            title="Testing Complete",
            text="Testing process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Testing Error",
            text="An error occurred during testing",
            level="ERROR",
        )
        raise e


def predict(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    predict_loader = setup.get_predict_loader()
    architecture = setup.get_architecture()
    callbacks = setup.get_callbacks()
    logger = setup.get_wandb_logger()

    logged_hparams = {}
    for key, value in config.architecture.items():
        if key != "_target_" and key != "model":
            logged_hparams[key] = value
    if "model" in config.architecture:
        for key, value in config.architecture["model"].items():
            if key != "_target_":
                logged_hparams[key] = value
    logged_hparams["batch_size"] = config.batch_size
    logged_hparams["epoch"] = config.epoch
    logged_hparams["seed"] = config.seed
    for key, value in config.trainer.items():
        if key != "_target_":
            logged_hparams[key] = value
    for key, value in config.dataset.items():
        if key not in [
            "_target_",
            "data_path",
            "split",
            "seed",
        ]:
            logged_hparams[key] = value
    logger.log_hyperparams(logged_hparams)

    trainer: Trainer = instantiate(
        config.trainer,
        strategy="auto",
        devices=1,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
    )

    try:
        if (
            config.strategy == "deepspeed_stage_3"
            or config.strategy == "deepspeed_stage_3_offload"
        ):
            preds = trainer.predict(
                model=architecture,
                dataloaders=predict_loader,
                ckpt_path=f"{config.ckpt_path}/model.pt",
            )
        else:
            preds = trainer.predict(
                model=architecture,
                dataloaders=predict_loader,
                ckpt_path=config.ckpt_path,
            )
        logger.experiment.alert(
            title="Predicting Complete",
            text="Predicting process has successfully finished.",
            level="INFO",
        )
    except Exception as e:
        logger.experiment.alert(
            title="Predicting Error",
            text="An error occurred during predicting",
            level="ERROR",
        )
        raise e

    all_predictions_with_indices = {}
    for pred in preds:
        all_predictions_with_indices.update(pred)
    all_predictions = [
        all_predictions_with_indices[key]
        for key in sorted(all_predictions_with_indices.keys())
    ]
    pred_df = pd.read_csv(
        f"{config.connected_dir}/data/{config.submission_file_name}.csv"
    )
    if len(all_predictions) < len(pred_df):
        raise ValueError(
            f"Length of all_predictions {all_predictions} is shorter than length of predict data {pred_df}."
        )
    if len(all_predictions) > len(pred_df):
        all_predictions = all_predictions[: len(pred_df)]
    pred_df[config.target_column_name] = all_predictions
    if not os.path.exists(f"{config.connected_dir}/submissions"):
        os.makedirs(
            f"{config.connected_dir}/submissions",
            exist_ok=True,
        )
    pred_df.to_csv(
        f"{config.connected_dir}/submissions/{config.submission_name}.csv",
        index=False,
    )


def tune(
    config: DictConfig,
) -> None:
    if "seed" in config:
        seed_everything(config.seed)

    setup = SetUp(config)

    train_loader = setup.get_train_loader()
    val_loader = setup.get_val_loader()
    logger = setup.get_wandb_logger()

    tuner: HuggingFaceTuner = instantiate(
        config.tuner,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )
    tuner()
