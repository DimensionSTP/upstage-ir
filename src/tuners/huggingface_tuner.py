from typing import Dict, Any
import os
import json

from torch.utils.data import DataLoader

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

from transformers import BitsAndBytesConfig
from peft import LoraConfig

from ..architectures.models.huggingface_model import HuggingFaceModel
from ..architectures.huggingface_architecture import HuggingFaceArchitecture


class HuggingFaceTuner:
    def __init__(
        self,
        hparams: Dict[str, Any],
        module_params: Dict[str, Any],
        direction: str,
        seed: int,
        num_trials: int,
        hparams_save_path: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        logger: WandbLogger,
    ) -> None:
        self.hparams = hparams
        self.module_params = module_params
        self.direction = direction
        self.seed = seed
        self.num_trials = num_trials
        self.hparams_save_path = hparams_save_path
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger

    def __call__(self) -> None:
        study = optuna.create_study(
            direction=self.direction,
            sampler=TPESampler(seed=self.seed),
            pruner=HyperbandPruner(),
        )
        study.optimize(
            self.optuna_objective,
            n_trials=self.num_trials,
        )
        trial = study.best_trial
        best_score = trial.value
        best_params = trial.params
        print(f"Best score : {best_score}")
        print(f"Parameters : {best_params}")

        if not os.path.exists(self.hparams_save_path):
            os.makedirs(
                self.hparams_save_path,
                exist_ok=True,
            )

        with open(f"{self.hparams_save_path}/best_params.json", "w") as json_file:
            json.dump(
                best_params,
                json_file,
            )

    def optuna_objective(
        self,
        trial: optuna.trial.Trial,
    ) -> float:
        seed_everything(self.seed)

        params = dict()
        params["seed"] = self.seed
        if self.hparams.pretrained_model_name:
            params["pretrained_model_name"] = trial.suggest_categorical(
                name="pretrained_model_name",
                choices=self.hparams.pretrained_model_name,
            )
        if self.hparams.lr:
            params["lr"] = trial.suggest_float(
                name="lr",
                low=self.hparams.lr.low,
                high=self.hparams.lr.high,
                log=self.hparams.lr.log,
            )
        if self.hparams.weight_decay:
            params["weight_decay"] = trial.suggest_float(
                name="weight_decay",
                low=self.hparams.weight_decay.low,
                high=self.hparams.weight_decay.high,
                log=self.hparams.weight_decay.log,
            )
        if self.hparams.warmup_ratio:
            params["warmup_ratio"] = trial.suggest_float(
                name="warmup_ratio",
                low=self.hparams.warmup_ratio.low,
                high=self.hparams.warmup_ratio.high,
                log=self.hparams.warmup_ratio.log,
            )
        if self.hparams.eta_min_ratio:
            params["eta_min_ratio"] = trial.suggest_float(
                name="eta_min_ratio",
                low=self.hparams.eta_min_ratio.low,
                high=self.hparams.eta_min_ratio.high,
                log=self.hparams.eta_min_ratio.log,
            )

        model = HuggingFaceModel(
            pretrained_model_name=params["pretrained_model_name"],
            is_preprocessed=self.module_params.is_preprocessed,
            custom_data_encoder_path=self.module_params.custom_data_encoder_path,
            left_padding=self.module_params.left_padding,
            merged_model_path=self.module_params.merged_model_path,
            precision=self.module_params.precision,
            mode=self.module_params.model_execution_mode,
            quantization_type=self.module_params.quantization_type,
            quantization_config=BitsAndBytesConfig(
                **self.module_params.quantization_config
            ),
            peft_type=self.module_params.peft_type,
            peft_config=LoraConfig(**self.module_params.peft_config),
        )
        architecture = HuggingFaceArchitecture(
            model=model,
            pretrained_model_name=params["pretrained_model_name"],
            is_preprocessed=self.module_params.is_preprocessed,
            custom_data_encoder_path=self.module_params.custom_data_encoder_path,
            left_padding=self.module_params.left_padding,
            strategy=self.module_params.strategy,
            lr=params["lr"],
            weight_decay=params["weight_decay"],
            warmup_ratio=params["warmup_ratio"],
            eta_min_ratio=params["eta_min_ratio"],
            interval=self.module_params.interval,
            options=self.module_params.options,
            target_max_length=self.module_params.target_max_length,
            target_min_length=self.module_params.target_min_length,
            per_device_save_path=self.module_params.per_device_save_path,
            target_column_name=self.module_params.target_column_name,
        )

        self.logger.log_hyperparams(params)
        callbacks = EarlyStopping(
            monitor=self.module_params.monitor,
            mode=self.module_params.mode,
            patience=self.module_params.patience,
            min_delta=self.module_params.min_delta,
        )

        trainer = Trainer(
            devices=self.module_params.devices,
            accelerator=self.module_params.accelerator,
            strategy=self.module_params.strategy,
            log_every_n_steps=self.module_params.log_every_n_steps,
            precision=self.module_params.precision,
            accumulate_grad_batches=self.module_params.accumulate_grad_batches,
            gradient_clip_val=self.module_params.gradient_clip_val,
            gradient_clip_algorithm=self.module_params.gradient_clip_algorithm,
            max_epochs=self.module_params.max_epochs,
            enable_checkpointing=False,
            callbacks=callbacks,
            logger=self.logger,
        )

        try:
            trainer.fit(
                model=architecture,
                train_dataloaders=self.train_loader,
                val_dataloaders=self.val_loader,
            )
            self.logger.experiment.alert(
                title="Tuning Complete",
                text="Tuning process has successfully finished.",
                level="INFO",
            )
        except Exception as e:
            self.logger.experiment.alert(
                title="Tuning Error",
                text="An error occurred during tuning",
                level="ERROR",
            )
            raise e

        return trainer.callback_metrics[self.module_params.monitor].item()
