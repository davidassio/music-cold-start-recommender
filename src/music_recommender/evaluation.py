"""Construct fixed-target cold-start evaluation scenarios."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

DEFAULT_INTERACTIONS_PATH = Path(
    "data/interim/modeling/distinct_interactions.parquet"
)
DEFAULT_SPLITS_PATH = Path(
    "data/interim/modeling/user_splits.parquet"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/interim/modeling"
)

SEED_SIZES = (1, 3, 5, 10)
PROFILE_POOL_SIZE = max(SEED_SIZES)
TARGET_WINDOW_SIZE = 10


def attach_splits(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach user-level split labels to interactions."""

    interactions = interactions.copy()
    assignments = assignments.copy()

    interactions["user"] = interactions["user"].astype(
        str
    )
    assignments["user"] = assignments["user"].astype(
        str
    )

    merged = interactions.merge(
        assignments,
        on="user",
        how="left",
        validate="many_to_one",
    )

    if merged["split"].isna().any():
        raise ValueError(
            "At least one interaction has no split assignment."
        )

    return merged


def build_training_catalog(
    split_interactions: pd.DataFrame,
) -> set[str]:
    """Return tracks observed among training users."""

    return set(
        split_interactions.loc[
            split_interactions["split"] == "train",
            "song",
        ].astype(str)
    )


def build_evaluation_histories(
    split_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Order validation and test histories chronologically."""

    evaluation = split_interactions[
        split_interactions["split"].isin(
            ["validation", "test"]
        )
    ].copy()

    evaluation["song"] = evaluation["song"].astype(str)

    evaluation = evaluation.sort_values(
        ["split", "user", "first_timestamp", "song"],
        kind="stable",
    ).reset_index(drop=True)

    evaluation["history_position"] = (
        evaluation.groupby(
            ["split", "user"],
            observed=True,
        )
        .cumcount()
        .add(1)
    )

    return evaluation


def build_profile_pool(
    evaluation: pd.DataFrame,
) -> pd.DataFrame:
    """Select the first ten tracks for each user profile."""

    profile = evaluation[
        evaluation["history_position"] <= PROFILE_POOL_SIZE
    ].copy()

    return profile[
        [
            "split",
            "user",
            "song",
            "first_timestamp",
            "history_position",
        ]
    ].rename(
        columns={"history_position": "profile_position"}
    )


def build_seed_scenarios(
    profile: pd.DataFrame,
) -> pd.DataFrame:
    """Reveal progressively larger subsets of each profile."""

    scenario_frames = []

    for seed_size in SEED_SIZES:
        scenario = profile[
            profile["profile_position"] <= seed_size
        ].copy()

        scenario["seed_size"] = seed_size

        scenario_frames.append(scenario)

    seeds = pd.concat(
        scenario_frames,
        ignore_index=True,
    )

    return seeds[
        [
            "split",
            "user",
            "seed_size",
            "song",
            "first_timestamp",
            "profile_position",
        ]
    ]


def build_fixed_targets(
    evaluation: pd.DataFrame,
    training_catalog: set[str],
) -> pd.DataFrame:
    """Use the next ten tracks as fixed targets."""

    target_start = PROFILE_POOL_SIZE + 1
    target_end = PROFILE_POOL_SIZE + TARGET_WINDOW_SIZE

    targets = evaluation[
        evaluation["history_position"].between(
            target_start,
            target_end,
        )
    ].copy()

    targets["target_position"] = (
        targets["history_position"]
        - PROFILE_POOL_SIZE
    )

    targets["in_training_catalog"] = targets[
        "song"
    ].isin(training_catalog)

    return targets[
        [
            "split",
            "user",
            "song",
            "first_timestamp",
            "target_position",
            "in_training_catalog",
        ]
    ]


def validate_evaluation_data(
    profile: pd.DataFrame,
    seeds: pd.DataFrame,
    targets: pd.DataFrame,
) -> None:
    """Verify profile, seed, and target invariants."""

    profile_counts = profile.groupby(
        ["split", "user"],
        observed=True,
    ).size()

    if not (profile_counts == PROFILE_POOL_SIZE).all():
        raise ValueError(
            "Each evaluation user must have ten profile tracks."
        )

    target_counts = targets.groupby(
        ["split", "user"],
        observed=True,
    ).size()

    if not (target_counts == TARGET_WINDOW_SIZE).all():
        raise ValueError(
            "Each evaluation user must have exactly ten targets."
        )

    seed_counts = (
        seeds.groupby(
            ["split", "user", "seed_size"],
            observed=True,
        )
        .size()
        .reset_index(name="observed_seed_count")
    )

    if not (
        seed_counts["seed_size"]
        == seed_counts["observed_seed_count"]
    ).all():
        raise ValueError(
            "At least one scenario has an incorrect seed count."
        )

    profile_pairs = set(
        zip(profile["user"], profile["song"])
    )
    target_pairs = set(
        zip(targets["user"], targets["song"])
    )

    if profile_pairs & target_pairs:
        raise ValueError(
            "Profile and target tracks overlap."
        )


def build_evaluation_summary(
    seeds: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize each cold-start evaluation scenario."""

    target_user_summary = (
        targets.groupby(
            ["split", "user"],
            observed=True,
        )
        .agg(
            target_tracks=("song", "size"),
            known_target_tracks=(
                "in_training_catalog",
                "sum",
            ),
        )
        .reset_index()
    )

    rows = []

    for split_name in ["validation", "test"]:
        split_targets = target_user_summary[
            target_user_summary["split"] == split_name
        ]

        total_targets = int(
            split_targets["target_tracks"].sum()
        )
        known_targets = int(
            split_targets["known_target_tracks"].sum()
        )

        for seed_size in SEED_SIZES:
            split_seeds = seeds[
                (seeds["split"] == split_name)
                & (seeds["seed_size"] == seed_size)
            ]

            rows.append(
                {
                    "split": split_name,
                    "seed_size": seed_size,
                    "users": split_seeds["user"].nunique(),
                    "seed_interactions": len(split_seeds),
                    "target_interactions": total_targets,
                    "median_targets_per_user": (
                        split_targets["target_tracks"].median()
                    ),
                    "train_catalog_target_pct": (
                        known_targets / total_targets * 100
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_evaluation_data(
    profile: pd.DataFrame,
    seeds: pd.DataFrame,
    targets: pd.DataFrame,
    summary: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save evaluation tables and their audit summary."""

    output_directory.mkdir(parents=True, exist_ok=True)

    profile.to_parquet(
        output_directory / "evaluation_profile.parquet",
        index=False,
    )
    seeds.to_parquet(
        output_directory / "evaluation_seeds.parquet",
        index=False,
    )
    targets.to_parquet(
        output_directory / "evaluation_targets.parquet",
        index=False,
    )
    summary.to_csv(
        output_directory / "evaluation_summary.csv",
        index=False,
    )


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Build cold-start evaluation scenarios."
    )
    parser.add_argument(
        "--interactions-path",
        type=Path,
        default=DEFAULT_INTERACTIONS_PATH,
    )
    parser.add_argument(
        "--splits-path",
        type=Path,
        default=DEFAULT_SPLITS_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    """Build and validate fixed-target evaluation scenarios."""

    args = parse_arguments()

    for path in [
        args.interactions_path,
        args.splits_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading filtered interactions and user splits...")
    interactions = pd.read_parquet(
        args.interactions_path
    )
    assignments = pd.read_parquet(args.splits_path)

    split_interactions = attach_splits(
        interactions,
        assignments,
    )

    training_catalog = build_training_catalog(
        split_interactions
    )

    evaluation = build_evaluation_histories(
        split_interactions
    )
    profile = build_profile_pool(evaluation)
    seeds = build_seed_scenarios(profile)
    targets = build_fixed_targets(
        evaluation,
        training_catalog,
    )

    validate_evaluation_data(
        profile,
        seeds,
        targets,
    )

    summary = build_evaluation_summary(
        seeds,
        targets,
    )

    save_evaluation_data(
        profile,
        seeds,
        targets,
        summary,
        args.output_directory,
    )

    print("\nCold-start evaluation summary")
    print(summary.to_string(index=False))

    print(
        "\nTraining catalog tracks:",
        f"{len(training_catalog):,}",
    )
    print(
        "Saved evaluation data to:",
        args.output_directory,
    )


if __name__ == "__main__":
    main()