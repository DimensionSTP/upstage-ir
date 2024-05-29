import dotenv

dotenv.load_dotenv(
    override=True,
)

import os
import pickle
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
    connected_dir = config.connected_dir
    voted_logit = config.voted_logit
    voted_prediction = config.voted_prediction
    votings = config.votings

    weights = list(votings.values())
    if not np.isclose(sum(weights), 1):
        raise ValueError(f"summation of weights({sum(weights)}) is not equal to 1")

    weighted_logits = None
    for logit_file, weight in votings.items():
        try:
            with open(
                f"{connected_dir}/logits/{logit_file}.pickle",
                "rb",
            ) as f:
                logit = pickle.load(f)
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
    with open(
        voted_logit,
        "wb",
    ) as f:
        pickle.dump(
            weighted_logits,
            f,
        )
    with open(
        voted_prediction,
        "wb",
    ) as f:
        pickle.dump(
            ensemble_predictions,
            f,
        )


if __name__ == "__main__":
    softly_vote_logits()
