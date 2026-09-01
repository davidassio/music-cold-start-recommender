"""Construct the filtered recommendation dataset."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from music_recommender.eda import load_listening_history

DEFAULT_HISTORY_PATH = Path(
    "data/raw/music4all/listening_history.csv"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/interim/modeling"
)

MINIMUM_USER_TRACKS = 20
MINIMUM_TRACK_LISTENERS = 5


def build_distinct_interactions(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Keep each user's first recorded interaction with each track."""

    distinct_interactions = (
        history.sort_values(
            ["user", "timestamp", "song"],
            kind="stable",
        )
        .drop_duplicates(
            subset=["user", "song"],
            keep="first",
        )
        .rename(columns={"timestamp": "first_timestamp"})
        .reset_index(drop=True)
    )

    return distinct_interactions


def summarize_interactions(
    interactions: pd.DataFrame,
) -> dict[str, int]:
    """Return the central interaction-table counts."""

    return {
        "interactions": len(interactions),
        "users": interactions["user"].nunique(),
        "tracks": interactions["song"].nunique(),
    }


def iteratively_filter_interactions(
    interactions: pd.DataFrame,
    minimum_user_tracks: int = MINIMUM_USER_TRACKS,
    minimum_track_listeners: int = MINIMUM_TRACK_LISTENERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter sparse users and tracks until the dataset stabilizes."""

    filtered = interactions.copy()
    audit_rows = []
    iteration = 0

    while True:
        iteration += 1
        before = summarize_interactions(filtered)

        user_track_counts = filtered.groupby(
            "user",
            observed=True,
        )["song"].transform("size")

        filtered = filtered[
            user_track_counts >= minimum_user_tracks
        ].copy()

        track_listener_counts = filtered.groupby(
            "song",
            observed=True,
        )["user"].transform("size")

        filtered = filtered[
            track_listener_counts >= minimum_track_listeners
        ].copy()

        after = summarize_interactions(filtered)

        audit_rows.append(
            {
                "iteration": iteration,
                "interactions_before": before["interactions"],
                "users_before": before["users"],
                "tracks_before": before["tracks"],
                "interactions_after": after["interactions"],
                "users_after": after["users"],
                "tracks_after": after["tracks"],
            }
        )

        if before == after:
            break

    filtered = filtered.sort_values(
        ["user", "first_timestamp", "song"],
        kind="stable",
    ).reset_index(drop=True)

    audit = pd.DataFrame(audit_rows)

    return filtered, audit


def validate_filtered_interactions(
    interactions: pd.DataFrame,
    minimum_user_tracks: int,
    minimum_track_listeners: int,
) -> None:
    """Verify the final table satisfies all modeling constraints."""

    if interactions.duplicated(["user", "song"]).any():
        raise ValueError(
            "Filtered data contains duplicate user-track pairs."
        )

    user_track_counts = interactions.groupby(
        "user",
        observed=True,
    ).size()

    track_listener_counts = interactions.groupby(
        "song",
        observed=True,
    ).size()

    if user_track_counts.min() < minimum_user_tracks:
        raise ValueError(
            "At least one user falls below the user threshold."
        )

    if track_listener_counts.min() < minimum_track_listeners:
        raise ValueError(
            "At least one track falls below the track threshold."
        )

    timestamps_are_ordered = (
        interactions.groupby(
            "user",
            observed=True,
        )["first_timestamp"]
        .apply(lambda values: values.is_monotonic_increasing)
        .all()
    )

    if not timestamps_are_ordered:
        raise ValueError(
            "At least one user's interactions are not chronological."
        )


def save_modeling_data(
    interactions: pd.DataFrame,
    audit: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save the filtered interactions and filtering audit."""

    output_directory.mkdir(parents=True, exist_ok=True)

    interactions.to_parquet(
        output_directory / "distinct_interactions.parquet",
        index=False,
    )

    audit.to_csv(
        output_directory / "filtering_audit.csv",
        index=False,
    )


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Build the filtered recommendation dataset."
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--minimum-user-tracks",
        type=int,
        default=MINIMUM_USER_TRACKS,
    )
    parser.add_argument(
        "--minimum-track-listeners",
        type=int,
        default=MINIMUM_TRACK_LISTENERS,
    )

    return parser.parse_args()


def main() -> None:
    """Build and validate the modeling dataset."""

    args = parse_arguments()

    if not args.history_path.exists():
        raise FileNotFoundError(args.history_path)

    print("Loading listening history...")
    history = load_listening_history(args.history_path)

    print("Keeping first interaction with each distinct track...")
    distinct_interactions = build_distinct_interactions(history)

    initial_summary = summarize_interactions(
        distinct_interactions
    )

    print("\nDistinct interaction summary")
    for name, value in initial_summary.items():
        print(f"{name}: {value:,}")

    print("\nApplying iterative filtering...")
    filtered_interactions, audit = (
        iteratively_filter_interactions(
            distinct_interactions,
            minimum_user_tracks=args.minimum_user_tracks,
            minimum_track_listeners=(
                args.minimum_track_listeners
            ),
        )
    )

    validate_filtered_interactions(
        filtered_interactions,
        minimum_user_tracks=args.minimum_user_tracks,
        minimum_track_listeners=args.minimum_track_listeners,
    )

    save_modeling_data(
        filtered_interactions,
        audit,
        args.output_directory,
    )

    final_summary = summarize_interactions(
        filtered_interactions
    )

    print("\nFiltering audit")
    print(audit.to_string(index=False))

    print("\nFinal modeling dataset")
    for name, value in final_summary.items():
        print(f"{name}: {value:,}")

    print(
        "\nSaved modeling data to:",
        args.output_directory,
    )


if __name__ == "__main__":
    main()