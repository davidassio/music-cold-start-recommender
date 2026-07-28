"""Audit the Music4All-Onion user-track interaction dataset."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/userid_trackid_count.tsv.bz2")
COLUMN_NAMES = ["user_id", "track_id", "count"]


def audit_interactions(
    file_path: Path,
    chunk_size: int = 250_000,
    max_chunks: int | None = None,
) -> None:
    """Calculate summary statistics without loading the entire file at once."""

    total_rows = 0
    total_plays = 0
    unique_users: set[str] = set()
    unique_tracks: set[str] = set()
    minimum_play_count: int | None = None
    maximum_play_count: int | None = None

    chunks = pd.read_csv(
    file_path,
    sep="\t",
    compression="bz2",
    header=None,
    skiprows=1,
    names=["user_id", "track_id", "count"],
    chunksize=chunk_size,
    dtype={
        "user_id": "string",
        "track_id": "string",
        "count": "int32",
    },
)

    for chunk_number, chunk in enumerate(chunks, start=1):
        if max_chunks is not None and chunk_number > max_chunks:
            break

        total_rows += len(chunk)
        total_plays += int(chunk["count"].sum())

        unique_users.update(chunk["user_id"].dropna().unique())
        unique_tracks.update(chunk["track_id"].dropna().unique())

        chunk_minimum = int(chunk["count"].min())
        chunk_maximum = int(chunk["count"].max())

        minimum_play_count = (
            chunk_minimum
            if minimum_play_count is None
            else min(minimum_play_count, chunk_minimum)
        )
        maximum_play_count = (
            chunk_maximum
            if maximum_play_count is None
            else max(maximum_play_count, chunk_maximum)
        )

        print(
            f"Processed chunk {chunk_number:,}: "
            f"{total_rows:,} rows scanned"
        )

    possible_user_track_pairs = len(unique_users) * len(unique_tracks)
    sparsity = (
        1 - total_rows / possible_user_track_pairs
        if possible_user_track_pairs
        else 0
    )

    print("\nAudit summary")
    print(f"Rows:                 {total_rows:,}")
    print(f"Unique users:         {len(unique_users):,}")
    print(f"Unique tracks:        {len(unique_tracks):,}")
    print(f"Total plays:          {total_plays:,}")
    print(f"Minimum play count:   {minimum_play_count}")
    print(f"Maximum play count:   {maximum_play_count}")
    print(f"Observed density:     {1 - sparsity:.6%}")
    print(f"Matrix sparsity:      {sparsity:.6%}")


def parse_arguments():
    """Read command-line options."""

    parser = ArgumentParser(
        description="Audit Music4All-Onion interaction data."
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the compressed interaction file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=250_000,
        help="Rows to process in memory at one time.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional number of chunks to process.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the interaction audit."""

    args = parse_arguments()

    if not args.file_path.exists():
        raise FileNotFoundError(
            f"Interaction file not found: {args.file_path}"
        )

    audit_interactions(
        file_path=args.file_path,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )


if __name__ == "__main__":
    main()