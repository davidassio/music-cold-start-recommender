"""Audit the Music4All-Onion genre TF-IDF dataset."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/id_genres_tf-idf.tsv.bz2")
DEFAULT_OUTPUT_DIRECTORY = Path("reports/tables")


def load_genre_data(file_path: Path) -> tuple[pd.DataFrame, str]:
    """Load the genre TF-IDF data using memory-efficient data types."""

    column_names = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        nrows=0,
    ).columns.tolist()

    if len(column_names) < 2:
        raise ValueError("Expected one ID column and at least one genre column.")

    id_column = column_names[0]
    genre_columns = column_names[1:]

    data_types = {
        id_column: "string",
        **{column: "float32" for column in genre_columns},
    }

    genre_data = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        dtype=data_types,
    )

    return genre_data, id_column


def build_dataset_summary(
    genre_data: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    """Calculate dataset-wide genre statistics."""

    genre_values = genre_data.drop(columns=id_column)
    nonzero_mask = genre_values.ne(0)

    total_feature_cells = genre_values.size
    nonzero_feature_cells = int(nonzero_mask.sum().sum())
    missing_feature_cells = int(genre_values.isna().sum().sum())
    negative_feature_cells = int(genre_values.lt(0).sum().sum())

    tracks_without_genres = int(nonzero_mask.sum(axis=1).eq(0).sum())
    feature_density = (
        nonzero_feature_cells / total_feature_cells
        if total_feature_cells
        else 0
    )

    summary = {
        "Tracks": len(genre_data),
        "Unique track IDs": genre_data[id_column].nunique(),
        "Duplicate track IDs": genre_data[id_column].duplicated().sum(),
        "Missing track IDs": genre_data[id_column].isna().sum(),
        "Genre features": genre_values.shape[1],
        "Total feature cells": total_feature_cells,
        "Nonzero feature cells": nonzero_feature_cells,
        "Missing feature cells": missing_feature_cells,
        "Negative feature cells": negative_feature_cells,
        "Tracks without active genres": tracks_without_genres,
        "Feature density": feature_density,
        "Feature sparsity": 1 - feature_density,
    }

    return pd.DataFrame(
        {
            "statistic": summary.keys(),
            "value": summary.values(),
        }
    )


def build_feature_catalog(genre_columns: list[str]) -> pd.DataFrame:
    """Create a numbered table containing every genre feature."""

    return pd.DataFrame(
        {
            "feature_number": range(1, len(genre_columns) + 1),
            "genre": genre_columns,
        }
    )


def build_feature_summary(
    genre_values: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize how frequently each genre feature appears."""

    nonzero_mask = genre_values.ne(0)
    tracks_with_feature = nonzero_mask.sum(axis=0)
    active_weight_sums = genre_values.where(nonzero_mask).sum(axis=0)

    mean_active_weight = active_weight_sums.div(
        tracks_with_feature.replace(0, np.nan)
    )

    feature_summary = pd.DataFrame(
        {
            "genre": genre_values.columns,
            "tracks_with_feature": tracks_with_feature.values,
            "track_coverage_pct": (
                tracks_with_feature.values / len(genre_values) * 100
            ),
            "mean_active_tfidf": mean_active_weight.values,
            "maximum_tfidf": genre_values.max(axis=0).values,
        }
    )

    return feature_summary.sort_values(
        "tracks_with_feature",
        ascending=False,
    ).reset_index(drop=True)


def build_track_summary(
    genre_data: pd.DataFrame,
    id_column: str,
) -> pd.DataFrame:
    """Summarize active genres and strongest genre for each track."""

    genre_values = genre_data.drop(columns=id_column)
    value_array = genre_values.to_numpy()
    active_genre_counts = np.count_nonzero(value_array, axis=1)

    strongest_positions = value_array.argmax(axis=1)
    strongest_weights = value_array[
        np.arange(len(value_array)),
        strongest_positions,
    ]

    genre_names = genre_values.columns.to_numpy()
    strongest_genres = genre_names[strongest_positions].astype(object)

    no_active_genres = active_genre_counts == 0
    strongest_genres[no_active_genres] = pd.NA
    strongest_weights[no_active_genres] = np.nan

    return pd.DataFrame(
        {
            "track_id": genre_data[id_column],
            "active_genre_count": active_genre_counts,
            "strongest_genre": strongest_genres,
            "strongest_tfidf": strongest_weights,
        }
    )


def save_audit_tables(
    genre_data: pd.DataFrame,
    id_column: str,
    output_directory: Path,
) -> None:
    """Build and save reusable genre audit tables."""

    output_directory.mkdir(parents=True, exist_ok=True)

    genre_values = genre_data.drop(columns=id_column)
    genre_columns = genre_values.columns.tolist()

    dataset_summary = build_dataset_summary(genre_data, id_column)
    feature_catalog = build_feature_catalog(genre_columns)
    feature_summary = build_feature_summary(genre_values)
    track_summary = build_track_summary(genre_data, id_column)

    dataset_summary.to_csv(
        output_directory / "genre_dataset_summary.csv",
        index=False,
    )
    feature_catalog.to_csv(
        output_directory / "genre_feature_catalog.csv",
        index=False,
    )
    feature_summary.to_csv(
        output_directory / "genre_feature_summary.csv",
        index=False,
    )
    track_summary.to_csv(
        output_directory / "genre_track_summary.csv",
        index=False,
    )

    print("\nGenre dataset summary")
    print(dataset_summary.to_string(index=False))

    print("\nTen most prevalent genre features")
    print(feature_summary.head(10).to_string(index=False))

    print(f"\nAudit tables saved to: {output_directory}")


def parse_arguments():
    """Read command-line options."""

    parser = ArgumentParser(
        description="Audit Music4All-Onion genre TF-IDF data."
    )
    parser.add_argument(
        "--file-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the compressed genre feature file.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for generated audit tables.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the genre audit."""

    args = parse_arguments()

    if not args.file_path.exists():
        raise FileNotFoundError(
            f"Genre file not found: {args.file_path}"
        )

    print(f"Loading genre features from: {args.file_path}")
    genre_data, id_column = load_genre_data(args.file_path)

    print(
        f"Loaded {len(genre_data):,} tracks and "
        f"{genre_data.shape[1] - 1:,} genre features."
    )

    save_audit_tables(
        genre_data=genre_data,
        id_column=id_column,
        output_directory=args.output_directory,
    )


if __name__ == "__main__":
    main()