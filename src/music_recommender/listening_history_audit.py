"""Audit the timestamped Music4All listening history."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import pandas as pd

DEFAULT_HISTORY_PATH = Path(
    "data/raw/music4all/listening_history.csv"
)
DEFAULT_INFORMATION_PATH = Path(
    "data/raw/music4all/id_information.csv"
)
DEFAULT_OUTPUT_PATH = Path(
    "reports/tables/listening_history_summary.csv"
)
DEFAULT_CHUNK_SIZE = 500_000


def load_metadata_track_ids(file_path: Path) -> set[str]:
    """Load the valid Music4All track IDs."""

    metadata = pd.read_csv(
        file_path,
        sep="\t",
        usecols=["id"],
        dtype={"id": "string"},
    )
    return set(metadata["id"].dropna())


def audit_listening_history(
    history_path: Path,
    metadata_track_ids: set[str],
    chunk_size: int,
) -> pd.DataFrame:
    """Audit listening events using bounded-memory processing."""

    user_event_counts: Counter[str] = Counter()
    track_event_counts: Counter[str] = Counter()
    history_track_ids: set[str] = set()

    total_events = 0
    missing_users = 0
    missing_tracks = 0
    missing_timestamps = 0
    invalid_timestamps = 0
    earliest_timestamp = None
    latest_timestamp = None

    chunks = pd.read_csv(
        history_path,
        sep="\t",
        dtype={
            "user": "string",
            "song": "string",
            "timestamp": "string",
        },
        chunksize=chunk_size,
    )

    for chunk_number, chunk in enumerate(chunks, start=1):
        total_events += len(chunk)

        missing_users += int(chunk["user"].isna().sum())
        missing_tracks += int(chunk["song"].isna().sum())
        missing_timestamps += int(chunk["timestamp"].isna().sum())

        valid_users = chunk["user"].dropna()
        valid_tracks = chunk["song"].dropna()

        user_event_counts.update(valid_users)
        track_event_counts.update(valid_tracks)
        history_track_ids.update(valid_tracks)

        parsed_timestamps = pd.to_datetime(
            chunk["timestamp"],
            errors="coerce",
        )

        invalid_timestamps += int(
            (
                chunk["timestamp"].notna()
                & parsed_timestamps.isna()
            ).sum()
        )

        chunk_earliest = parsed_timestamps.min()
        chunk_latest = parsed_timestamps.max()

        if pd.notna(chunk_earliest):
            earliest_timestamp = (
                chunk_earliest
                if earliest_timestamp is None
                else min(earliest_timestamp, chunk_earliest)
            )

        if pd.notna(chunk_latest):
            latest_timestamp = (
                chunk_latest
                if latest_timestamp is None
                else max(latest_timestamp, chunk_latest)
            )

        print(
            f"Processed chunk {chunk_number}: "
            f"{total_events:,} events"
        )

    shared_tracks = history_track_ids & metadata_track_ids
    history_only_tracks = history_track_ids - metadata_track_ids

    user_counts = pd.Series(
        list(user_event_counts.values()),
        dtype="int64",
    )
    track_counts = pd.Series(
        list(track_event_counts.values()),
        dtype="int64",
    )

    summary = {
        "Listening events": total_events,
        "Unique users": len(user_event_counts),
        "Unique tracks": len(history_track_ids),
        "Tracks with metadata": len(shared_tracks),
        "Tracks without metadata": len(history_only_tracks),
        "Metadata coverage": (
            len(shared_tracks) / len(history_track_ids)
            if history_track_ids
            else 0
        ),
        "Missing users": missing_users,
        "Missing tracks": missing_tracks,
        "Missing timestamps": missing_timestamps,
        "Invalid timestamps": invalid_timestamps,
        "Earliest timestamp": earliest_timestamp,
        "Latest timestamp": latest_timestamp,
        "Minimum events per user": user_counts.min(),
        "Median events per user": user_counts.median(),
        "Mean events per user": user_counts.mean(),
        "Maximum events per user": user_counts.max(),
        "Minimum events per track": track_counts.min(),
        "Median events per track": track_counts.median(),
        "Mean events per track": track_counts.mean(),
        "Maximum events per track": track_counts.max(),
    }

    return pd.DataFrame(
        {
            "statistic": summary.keys(),
            "value": summary.values(),
        }
    )


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Audit timestamped Music4All listening history."
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
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )
    return parser.parse_args()


def main() -> None:
    """Run the listening-history audit."""

    args = parse_arguments()

    if not args.history_path.exists():
        raise FileNotFoundError(args.history_path)

    if not args.information_path.exists():
        raise FileNotFoundError(args.information_path)

    metadata_track_ids = load_metadata_track_ids(
        args.information_path
    )

    summary = audit_listening_history(
        history_path=args.history_path,
        metadata_track_ids=metadata_track_ids,
        chunk_size=args.chunk_size,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_path, index=False)

    print("\nListening-history summary")
    print(summary.to_string(index=False))

    print(f"\nSummary saved to: {args.output_path}")


if __name__ == "__main__":
    main()