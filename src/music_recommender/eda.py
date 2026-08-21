"""Build reusable EDA tables from the Music4All listening history."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_HISTORY_PATH = Path("data/raw/music4all/listening_history.csv")
DEFAULT_INFORMATION_PATH = Path("data/raw/music4all/id_information.csv")
DEFAULT_OUTPUT_DIRECTORY = Path("data/interim/eda")


def load_listening_history(file_path: Path) -> pd.DataFrame:
    """Load, validate, and deduplicate the listening history."""

    history = pd.read_csv(
        file_path,
        sep="\t",
        dtype={
            "user": "category",
            "song": "category",
        },
        parse_dates=["timestamp"],
    )

    expected_columns = {"user", "song", "timestamp"}
    missing_columns = expected_columns - set(history.columns)

    if missing_columns:
        raise ValueError(
            f"Listening history is missing columns: {sorted(missing_columns)}"
        )

    duplicate_count = int(history.duplicated().sum())

    history = history.drop_duplicates().reset_index(drop=True)
    history.attrs["exact_duplicates_removed"] = duplicate_count

    print(f"Removed {duplicate_count:,} exact duplicate events.")

    return history


def load_track_information(file_path: Path) -> pd.DataFrame:
    """Load readable track information."""

    information = pd.read_csv(
        file_path,
        sep="\t",
        dtype="string",
        usecols=["id", "artist", "song", "album_name"],
    )

    if information["id"].duplicated().any():
        raise ValueError("Track information contains duplicate IDs.")

    return information


def build_user_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Summarize listening activity and chronology for each user."""

    user_summary = (
        history.groupby("user", observed=True)
        .agg(
            listening_events=("song", "size"),
            distinct_tracks=("song", "nunique"),
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
        )
        .reset_index()
    )

    user_summary["repeat_events"] = (
        user_summary["listening_events"] - user_summary["distinct_tracks"]
    )
    user_summary["repeat_event_rate"] = (
        user_summary["repeat_events"] / user_summary["listening_events"]
    )
    user_summary["history_span_days"] = (
        user_summary["last_event"] - user_summary["first_event"]
    ).dt.total_seconds() / 86_400

    return user_summary.sort_values(
        "listening_events",
        ascending=False,
    ).reset_index(drop=True)


def build_track_summary(
    history: pd.DataFrame,
    information: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize track popularity and attach readable metadata."""

    track_summary = (
        history.groupby("song", observed=True)
        .agg(
            listening_events=("user", "size"),
            unique_listeners=("user", "nunique"),
            first_event=("timestamp", "min"),
            last_event=("timestamp", "max"),
        )
        .reset_index()
        .rename(columns={"song": "track_id"})
    )

    track_summary["events_per_listener"] = (
        track_summary["listening_events"]
        / track_summary["unique_listeners"]
    )

    track_summary = track_summary.merge(
        information,
        left_on="track_id",
        right_on="id",
        how="left",
        validate="one_to_one",
    ).drop(columns="id")

    track_summary["catalog_rank"] = (
        track_summary["listening_events"]
        .rank(method="first", ascending=False)
        .astype("int64")
    )

    return track_summary.sort_values(
        "catalog_rank"
    ).reset_index(drop=True)


def build_dataset_summary(
    history: pd.DataFrame,
    user_summary: pd.DataFrame,
    track_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate high-level EDA statistics."""

    duplicates_removed = history.attrs.get(
        "exact_duplicates_removed",
        0,
    )

    summary = {
        "Raw listening events": len(history) + duplicates_removed,
        "Exact duplicate events removed": duplicates_removed,
        "Clean listening events": len(history),
        "Unique users": history["user"].nunique(),
        "Unique tracks": history["song"].nunique(),
        "Earliest timestamp": history["timestamp"].min(),
        "Latest timestamp": history["timestamp"].max(),
        "Median events per user": user_summary["listening_events"].median(),
        "Median distinct tracks per user": (
            user_summary["distinct_tracks"].median()
        ),
        "Median repeat-event rate": (
            user_summary["repeat_event_rate"].median()
        ),
        "Median listeners per track": (
            track_summary["unique_listeners"].median()
        ),
        "Median events per track": (
            track_summary["listening_events"].median()
        ),
    }

    return pd.DataFrame(
        {
            "statistic": summary.keys(),
            "value": summary.values(),
        }
    )


def build_popularity_curve(
    track_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the cumulative catalog-popularity curve."""

    popularity = track_summary[
        [
            "catalog_rank",
            "track_id",
            "artist",
            "song",
            "listening_events",
            "unique_listeners",
        ]
    ].copy()

    popularity["catalog_fraction"] = (
        popularity["catalog_rank"] / len(popularity)
    )
    popularity["cumulative_event_fraction"] = (
        popularity["listening_events"].cumsum()
        / popularity["listening_events"].sum()
    )

    return popularity


def build_popularity_concentration(
    popularity_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Measure the event share captured by popular catalog segments."""

    rows = []

    for catalog_pct in [0.1, 0.5, 1, 5, 10, 20, 50]:
        catalog_fraction = catalog_pct / 100
        track_count = max(
            1,
            int(np.ceil(len(popularity_curve) * catalog_fraction)),
        )
        event_fraction = popularity_curve.iloc[
            track_count - 1
        ]["cumulative_event_fraction"]

        rows.append(
            {
                "top_catalog_pct": catalog_pct,
                "track_count": track_count,
                "event_share_pct": float(event_fraction * 100),
            }
        )

    return pd.DataFrame(rows)


def build_user_threshold_summary(
    user_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Show how cold-start eligibility changes by history threshold."""

    rows = []

    for minimum_tracks in [2, 5, 10, 20, 30, 50, 100]:
        eligible = user_summary[
            user_summary["distinct_tracks"] >= minimum_tracks
        ]

        rows.append(
            {
                "minimum_distinct_tracks": minimum_tracks,
                "eligible_users": len(eligible),
                "eligible_user_pct": (
                    len(eligible) / len(user_summary) * 100
                ),
                "retained_events": int(
                    eligible["listening_events"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_track_threshold_summary(
    track_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Show catalog retention at different listener thresholds."""

    rows = []

    for minimum_listeners in [1, 2, 5, 10, 20, 50, 100]:
        eligible = track_summary[
            track_summary["unique_listeners"] >= minimum_listeners
        ]

        rows.append(
            {
                "minimum_unique_listeners": minimum_listeners,
                "eligible_tracks": len(eligible),
                "eligible_track_pct": (
                    len(eligible) / len(track_summary) * 100
                ),
                "retained_events": int(
                    eligible["listening_events"].sum()
                ),
                "retained_event_pct": (
                    eligible["listening_events"].sum()
                    / track_summary["listening_events"].sum()
                    * 100
                ),
            }
        )

    return pd.DataFrame(rows)


def save_eda_tables(
    output_directory: Path,
    dataset_summary: pd.DataFrame,
    user_summary: pd.DataFrame,
    track_summary: pd.DataFrame,
    popularity_curve: pd.DataFrame,
    popularity_concentration: pd.DataFrame,
    user_thresholds: pd.DataFrame,
    track_thresholds: pd.DataFrame,
) -> None:
    """Save reusable EDA tables."""

    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_summary.to_csv(
        output_directory / "dataset_summary.csv",
        index=False,
    )
    popularity_concentration.to_csv(
        output_directory / "popularity_concentration.csv",
        index=False,
    )
    user_thresholds.to_csv(
        output_directory / "user_thresholds.csv",
        index=False,
    )
    track_thresholds.to_csv(
        output_directory / "track_thresholds.csv",
        index=False,
    )

    user_summary.to_parquet(
        output_directory / "user_summary.parquet",
        index=False,
    )
    track_summary.to_parquet(
        output_directory / "track_summary.parquet",
        index=False,
    )
    popularity_curve.to_parquet(
        output_directory / "popularity_curve.parquet",
        index=False,
    )


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Build Music4All EDA tables."
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
    )
    parser.add_argument(
        "--information-path",
        type=Path,
        default=DEFAULT_INFORMATION_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    """Run the EDA pipeline."""

    args = parse_arguments()

    for file_path in [args.history_path, args.information_path]:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

    print("Loading timestamped listening history...")
    history = load_listening_history(args.history_path)
    print(f"Loaded {len(history):,} listening events.")

    print("Loading track information...")
    information = load_track_information(args.information_path)

    print("Building user activity summary...")
    user_summary = build_user_summary(history)

    print("Building track popularity summary...")
    track_summary = build_track_summary(
        history,
        information,
    )

    dataset_summary = build_dataset_summary(
        history,
        user_summary,
        track_summary,
    )
    popularity_curve = build_popularity_curve(track_summary)
    popularity_concentration = build_popularity_concentration(
        popularity_curve
    )
    user_thresholds = build_user_threshold_summary(user_summary)
    track_thresholds = build_track_threshold_summary(track_summary)

    save_eda_tables(
        output_directory=args.output_directory,
        dataset_summary=dataset_summary,
        user_summary=user_summary,
        track_summary=track_summary,
        popularity_curve=popularity_curve,
        popularity_concentration=popularity_concentration,
        user_thresholds=user_thresholds,
        track_thresholds=track_thresholds,
    )

    print("\nDataset summary")
    print(dataset_summary.to_string(index=False))

    print("\nPopularity concentration")
    print(popularity_concentration.to_string(index=False))

    print("\nUser filtering thresholds")
    print(user_thresholds.to_string(index=False))

    print("\nTrack filtering thresholds")
    print(track_thresholds.to_string(index=False))

    print(f"\nEDA tables saved to: {args.output_directory}")


if __name__ == "__main__":
    main()