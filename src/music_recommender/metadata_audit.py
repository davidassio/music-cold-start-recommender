"""Audit the protected Music4All metadata tables."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

DEFAULT_DATA_DIRECTORY = Path("data/raw/music4all")
DEFAULT_OUTPUT_DIRECTORY = Path("reports/tables")

METADATA_FILES = [
    "id_information.csv",
    "id_metadata.csv",
    "id_genres.csv",
    "id_tags.csv",
    "id_lang.csv",
]


def load_table(file_path: Path) -> pd.DataFrame:
    """Load a tab-separated metadata table as strings."""

    return pd.read_csv(
        file_path,
        sep="\t",
        dtype="string",
    )


def summarize_table(
    file_name: str,
    table: pd.DataFrame,
) -> dict[str, object]:
    """Calculate structural statistics for one metadata table."""

    if "id" not in table.columns:
        raise ValueError(f"{file_name} does not contain an 'id' column.")

    return {
        "file": file_name,
        "rows": len(table),
        "columns": table.shape[1],
        "unique_track_ids": table["id"].nunique(),
        "duplicate_track_ids": int(table["id"].duplicated().sum()),
        "missing_track_ids": int(table["id"].isna().sum()),
        "missing_cells": int(table.isna().sum().sum()),
        "duplicate_rows": int(table.duplicated().sum()),
    }


def build_coverage_summary(
    tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Measure ID coverage relative to the information table."""

    reference_ids = set(tables["id_information.csv"]["id"].dropna())
    rows = []

    for file_name, table in tables.items():
        track_ids = set(table["id"].dropna())
        shared_ids = reference_ids & track_ids

        rows.append(
            {
                "file": file_name,
                "track_ids": len(track_ids),
                "shared_with_information": len(shared_ids),
                "information_coverage_pct": (
                    len(shared_ids) / len(reference_ids) * 100
                    if reference_ids
                    else 0
                ),
                "ids_not_in_information": len(track_ids - reference_ids),
                "information_ids_missing": len(reference_ids - track_ids),
            }
        )

    return pd.DataFrame(rows)


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Audit Music4All metadata tables."
    )
    parser.add_argument(
        "--data-directory",
        type=Path,
        default=DEFAULT_DATA_DIRECTORY,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    return parser.parse_args()


def main() -> None:
    """Run the metadata audit."""

    args = parse_arguments()
    tables = {}
    table_summaries = []

    for file_name in METADATA_FILES:
        file_path = args.data_directory / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        table = load_table(file_path)
        tables[file_name] = table
        table_summaries.append(summarize_table(file_name, table))

    table_summary = pd.DataFrame(table_summaries)
    coverage_summary = build_coverage_summary(tables)

    args.output_directory.mkdir(parents=True, exist_ok=True)

    table_summary.to_csv(
        args.output_directory / "metadata_table_summary.csv",
        index=False,
    )
    coverage_summary.to_csv(
        args.output_directory / "metadata_coverage_summary.csv",
        index=False,
    )

    print("\nMetadata table summary")
    print(table_summary.to_string(index=False))

    print("\nCoverage relative to id_information.csv")
    print(coverage_summary.to_string(index=False))

    information = tables["id_information.csv"]
    metadata = tables["id_metadata.csv"]

    preview = information.merge(
        metadata[["id", "spotify_id", "popularity", "release"]],
        on="id",
        how="left",
        validate="one_to_one",
    )

    print("\nExample recognizable tracks")
    print(
        preview[
            [
                "artist",
                "song",
                "album_name",
                "release",
                "popularity",
                "spotify_id",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print(f"\nSummary tables saved to: {args.output_directory}")


if __name__ == "__main__":
    main()