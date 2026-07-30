"""Measure track coverage between interaction and genre datasets."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

DEFAULT_INTERACTION_PATH = Path(
    "data/raw/userid_trackid_count.tsv.bz2"
)
DEFAULT_GENRE_PATH = Path("data/raw/id_genres_tf-idf.tsv.bz2")
DEFAULT_OUTPUT_PATH = Path("reports/tables/track_coverage_summary.csv")
DEFAULT_CHUNK_SIZE = 250_000


def load_genre_track_ids(file_path: Path) -> set[str]:
    """Load track IDs from the genre feature file."""

    genre_ids = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        usecols=[0],
        dtype="string",
    )

    id_column = genre_ids.columns[0]
    return set(genre_ids[id_column].dropna())


def load_interaction_track_ids(
    file_path: Path,
    chunk_size: int,
) -> set[str]:
    """Collect unique track IDs from the interaction file."""

    interaction_track_ids: set[str] = set()

    chunks = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        header=None,
        skiprows=1,
        names=["user_id", "track_id", "count"],
        usecols=["track_id"],
        dtype={"track_id": "string"},
        chunksize=chunk_size,
    )

    rows_scanned = 0

    for chunk_number, chunk in enumerate(chunks, start=1):
        interaction_track_ids.update(chunk["track_id"].dropna())
        rows_scanned += len(chunk)

        print(
            f"Processed chunk {chunk_number}: "
            f"{rows_scanned:,} rows scanned, "
            f"{len(interaction_track_ids):,} unique tracks found"
        )

    return interaction_track_ids


def build_coverage_summary(
    interaction_track_ids: set[str],
    genre_track_ids: set[str],
) -> pd.DataFrame:
    """Calculate track overlap and coverage statistics."""

    shared_track_ids = interaction_track_ids & genre_track_ids
    interaction_only_ids = interaction_track_ids - genre_track_ids
    genre_only_ids = genre_track_ids - interaction_track_ids

    interaction_coverage = (
        len(shared_track_ids) / len(interaction_track_ids)
        if interaction_track_ids
        else 0
    )
    genre_coverage = (
        len(shared_track_ids) / len(genre_track_ids)
        if genre_track_ids
        else 0
    )

    summary = {
        "Interaction tracks": len(interaction_track_ids),
        "Genre tracks": len(genre_track_ids),
        "Shared tracks": len(shared_track_ids),
        "Interaction tracks without genres": len(interaction_only_ids),
        "Genre tracks without interactions": len(genre_only_ids),
        "Interaction track coverage": interaction_coverage,
        "Genre track coverage": genre_coverage,
    }

    return pd.DataFrame(
        {
            "statistic": summary.keys(),
            "value": summary.values(),
        }
    )


def parse_arguments():
    """Read command-line options."""

    parser = ArgumentParser(
        description="Measure interaction and genre track coverage."
    )
    parser.add_argument(
        "--interaction-path",
        type=Path,
        default=DEFAULT_INTERACTION_PATH,
    )
    parser.add_argument(
        "--genre-path",
        type=Path,
        default=DEFAULT_GENRE_PATH,
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
    """Run the track coverage audit."""

    args = parse_arguments()

    if not args.interaction_path.exists():
        raise FileNotFoundError(
            f"Interaction file not found: {args.interaction_path}"
        )

    if not args.genre_path.exists():
        raise FileNotFoundError(
            f"Genre file not found: {args.genre_path}"
        )

    print("Loading genre track IDs...")
    genre_track_ids = load_genre_track_ids(args.genre_path)
    print(f"Found {len(genre_track_ids):,} unique genre tracks.")

    print("\nScanning interaction track IDs...")
    interaction_track_ids = load_interaction_track_ids(
        args.interaction_path,
        args.chunk_size,
    )

    coverage_summary = build_coverage_summary(
        interaction_track_ids,
        genre_track_ids,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_summary.to_csv(args.output_path, index=False)

    print("\nTrack coverage summary")
    print(coverage_summary.to_string(index=False))

    print(f"\nCoverage summary saved to: {args.output_path}")


if __name__ == "__main__":
    main()