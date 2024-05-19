import os
import warnings

os.environ["HYDRA_FULL_ERROR"] = "1"
warnings.filterwarnings("ignore")

import numpy as np

import hydra
from omegaconf import DictConfig


@hydra.main(
    config_path="configs/",
    config_name="voting.yaml",
)
def softly_vote_logits(
    config: DictConfig,
) -> None:
    basic_path = config.basic_path
    voted_logit = config.voted_logit
    voted_prediction = config.voted_prediction
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_logits = None
    for logit_file, weight in votings.items():
        try:
            logit = np.load(f"{basic_path}/logits/{logit_file}.npy")
        except:
            raise FileNotFoundError(f"logit file {logit_file} does not exist")
        if weighted_logits is None:
            weighted_logits = logit * weight
        else:
            weighted_logits += logit * weight

    ensemble_predictions = np.argmax(
        weighted_logits,
        axis=-1,
    )
    np.save(
        voted_logit,
        weighted_logits,
    )
    np.save(
        voted_prediction,
        ensemble_predictions,
    )


if __name__ == "__main__":
    softly_vote_logits()
