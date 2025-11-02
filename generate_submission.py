"""Generate top-1 book recommendations for each user.

This utility loads a trained NCL checkpoint from the ``saved`` directory, runs
the model in inference mode and produces a ``submission.csv`` file that
contains the top-1 recommendation for every user that appears in the
interaction log.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.utils import init_seed

from ncl import NCL


def _load_users_from_interactions(interactions_path: str) -> List[Tuple[object, str]]:
    """Return the distinct users found in the interaction log.

    The tuples contain the original value from the CSV as well as its string
    representation, which is used to look up internal identifiers in the
    RecBole dataset.
    """

    df = pd.read_csv(interactions_path)
    if "user_id" not in df.columns:
        raise ValueError("Interaction file must contain a 'user_id' column")

    # ``dict.fromkeys`` preserves the original order of appearance.
    unique_values: Sequence[object] = list(dict.fromkeys(df["user_id"].dropna().tolist()))
    return [(value, str(value)) for value in unique_values]


def _build_config(dataset: str, extra_config: Optional[Iterable[str]] = None) -> Config:
    """Create the RecBole configuration used for loading the checkpoint."""

    config_files = [
        os.path.join("properties", "overall.yaml"),
        os.path.join("properties", "NCL.yaml"),
    ]

    dataset_config = os.path.join("properties", f"{dataset}.yaml")
    if os.path.exists(dataset_config):
        config_files.append(dataset_config)

    if extra_config:
        config_files.extend(extra_config)

    config = Config(model=NCL, dataset=dataset, config_file_list=config_files)
    init_seed(config["seed"], config["reproducibility"])
    return config


def _load_model(config: Config, checkpoint_path: str) -> NCL:
    """Instantiate the NCL model and restore the saved parameters."""

    dataset = create_dataset(config)
    model = NCL(config, dataset).to(config["device"])

    checkpoint = torch.load(checkpoint_path, map_location=config["device"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _recommend_top1(model: NCL, user_values: Sequence[Tuple[object, str]]) -> List[Tuple[object, str]]:
    """Compute the top-1 item recommendation for each provided user."""

    dataset = model.dataset
    token2id = dataset.field2token_id[dataset.uid_field]
    id2token = dataset.field2id_token[dataset.iid_field]

    interaction_matrix = model.interaction_matrix.tocsr()

    with torch.no_grad():
        user_embeddings, item_embeddings, _ = model.forward()
    user_embeddings = user_embeddings.detach()
    item_embeddings = item_embeddings.detach()

    device = user_embeddings.device

    recommendations: List[Tuple[object, str]] = []
    for original_value, token in user_values:
        internal_user_id = token2id.get(token)
        if internal_user_id is None:
            logging.warning("User %s not found in the dataset; skipping.", token)
            continue

        user_vector = user_embeddings[internal_user_id]
        scores = torch.matmul(item_embeddings, user_vector)

        history_items = interaction_matrix.getrow(internal_user_id).indices
        if history_items.size:
            scores[torch.as_tensor(history_items, device=device)] = float("-inf")

        if not torch.isfinite(scores).any():
            logging.warning(
                "User %s has interacted with all available items; skipping.", token
            )
            continue

        top_item_index = torch.argmax(scores).item()
        top_item_token = id2token[top_item_index]
        recommendations.append((original_value, top_item_token))

    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate top-1 book recommendations.")
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model-path",
        type=str,
        help="Full path to the trained model checkpoint.",
    )
    model_group.add_argument(
        "--model-name",
        type=str,
        help="File name of a checkpoint stored inside --saved-dir (e.g. 'NCL-epoch-10.pth').",
    )
    parser.add_argument(
        "--saved-dir",
        type=str,
        default="saved",
        help="Directory that contains saved model checkpoints when using --model-name.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data",
        help="Name of the dataset used during training.",
    )
    parser.add_argument(
        "--interactions",
        type=str,
        default=os.path.join("data", "inter_preliminary.csv"),
        help="CSV file containing the raw interaction records.",
    )
    parser.add_argument(
        "--extra-config",
        type=str,
        nargs="*",
        help="Optional additional RecBole config files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Path of the generated submission file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (e.g., INFO, DEBUG).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    checkpoint_path = args.model_path
    if args.model_name:
        checkpoint_path = os.path.join(args.saved_dir, args.model_name)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    logging.info("Loading checkpoint from %s", checkpoint_path)

    user_values = _load_users_from_interactions(args.interactions)
    if not user_values:
        raise ValueError("No users found in the provided interaction file.")

    config = _build_config(args.dataset, args.extra_config)
    model = _load_model(config, checkpoint_path)

    recommendations = _recommend_top1(model, user_values)
    if not recommendations:
        raise RuntimeError("No recommendations could be generated.")

    submission_df = pd.DataFrame(recommendations, columns=["user_id", "item_id"])

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    submission_df.to_csv(args.output, index=False)

    logging.info("Saved %d recommendations to %s", len(submission_df), args.output)


if __name__ == "__main__":
    main()
