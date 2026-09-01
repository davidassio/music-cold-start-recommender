"""Create reproducible user-level modeling splits."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_INTERACTIONS_PATH = Path(
    "data/interim/modeling/distinct_interactions.parquet"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/interim/modeling"
)

TRAIN_FRACTION = 0.80
VALIDATION_FRACTION = 0.10
TEST_FRACTION = 0.10
RANDOM_SEED = 42


def assign_user_splits(
    interactions: pd.DataFrame,
    train_fraction: float = TRAIN_FRACTION,
    validation_fraction: float = VALIDATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Assign each user to exactly one modeling split."""

    fractions = (
        train_fraction,
        validation_fraction,
        test_fraction,
    )

    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Split fractions must sum to 1.")

    users = np.sort(
        interactions["user"].astype(str).unique()
    )

    generator = np.random.default_rng(random_seed)
    shuffled_users = generator.permutation(users)

    user_count = len(shuffled_users)
    validation_count = round(
        user_count * validation_fraction
    )
    test_count = round(user_count * test_fraction)
    train_count = (
        user_count
        - validation_count
        - test_count
    )

    split_labels = np.concatenate(
        [
            np.repeat("train", train_count),
            np.repeat("validation", validation_count),
            np.repeat("test", test_count),
        ]
    )

    assignments = pd.DataFrame(
        {
            "user": shuffled_users,
            "split": split_labels,
        }
    )

    return assignments.sort_values("user").reset_index(
        drop=True
    )


def validate_user_splits(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> None:
    """Verify that assignments are complete and disjoint."""

    if assignments["user"].duplicated().any():
        raise ValueError(
            "At least one user has multiple split assignments."
        )

    expected_splits = {"train", "validation", "test"}
    observed_splits = set(assignments["split"].unique())

    if observed_splits != expected_splits:
        raise ValueError(
            f"Unexpected split labels: {observed_splits}"
        )

    interaction_users = set(
        interactions["user"].astype(str).unique()
    )
    assigned_users = set(assignments["user"])

    if interaction_users != assigned_users:
        raise ValueError(
            "User assignments do not match interaction users."
        )


def attach_user_splits(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach split labels to the interaction table."""

    split_interactions = interactions.copy()
    split_interactions["user"] = (
        split_interactions["user"].astype(str)
    )

    split_interactions = split_interactions.merge(
        assignments,
        on="user",
        how="left",
        validate="many_to_one",
    )

    if split_interactions["split"].isna().any():
        raise ValueError(
            "At least one interaction has no split assignment."
        )

    return split_interactions


def build_split_summary(
    split_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize users, interactions, and catalog overlap."""

    training_tracks = set(
        split_interactions.loc[
            split_interactions["split"] == "train",
            "song",
        ]
    )

    total_users = split_interactions["user"].nunique()
    total_interactions = len(split_interactions)

    rows = []

    for split_name in ["train", "validation", "test"]:
        subset = split_interactions[
            split_interactions["split"] == split_name
        ]

        known_catalog_interactions = subset[
            "song"
        ].isin(training_tracks)

        rows.append(
            {
                "split": split_name,
                "users": subset["user"].nunique(),
                "user_pct": (
                    subset["user"].nunique()
                    / total_users
                    * 100
                ),
                "interactions": len(subset),
                "interaction_pct": (
                    len(subset)
                    / total_interactions
                    * 100
                ),
                "unique_tracks": subset["song"].nunique(),
                "train_catalog_interaction_pct": (
                    known_catalog_interactions.mean()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def save_split_data(
    assignments: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save user assignments and the split audit."""

    output_directory.mkdir(parents=True, exist_ok=True)

    assignments.to_parquet(
        output_directory / "user_splits.parquet",
        index=False,
    )

    summary.to_csv(
        output_directory / "split_summary.csv",
        index=False,
    )


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Create user-level modeling splits."
    )
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=DEFAULT_INTERACTIONS_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
    )

    return parser.parse_args()


def main() -> None:
    """Create, validate, and save the user splits."""

    args = parse_arguments()

    if not args.interactions_path.exists():
        raise FileNotFoundError(args.interactions_path)

    print("Loading filtered distinct interactions...")
    interactions = pd.read_parquet(
        args.interactions_path
    )

    print("Assigning users to modeling splits...")
    assignments = assign_user_splits(
        interactions,
        random_seed=args.random_seed,
    )

    validate_user_splits(
        interactions,
        assignments,
    )

    split_interactions = attach_user_splits(
        interactions,
        assignments,
    )

    summary = build_split_summary(
        split_interactions
    )

    save_split_data(
        assignments,
        summary,
        args.output_directory,
    )

    print("\nUser-level split summary")
    print(summary.to_string(index=False))

    print(
        "\nSaved split assignments to:",
        args.output_directory,
    )


if __name__ == "__main__":
    main()