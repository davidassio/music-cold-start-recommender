"""Fit and evaluate popularity and genre-content baselines."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

DEFAULT_INTERACTIONS_PATH = Path(
    "data/interim/modeling/distinct_interactions.parquet"
)
DEFAULT_SPLITS_PATH = Path(
    "data/interim/modeling/user_splits.parquet"
)
DEFAULT_PROFILE_PATH = Path(
    "data/interim/modeling/evaluation_profile.parquet"
)
DEFAULT_SEEDS_PATH = Path(
    "data/interim/modeling/evaluation_seeds.parquet"
)
DEFAULT_TARGETS_PATH = Path(
    "data/interim/modeling/evaluation_targets.parquet"
)
DEFAULT_GENRES_PATH = Path(
    "data/raw/id_genres_tf-idf.tsv.bz2"
)
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/interim/results"
)

SEED_SIZES = (1, 3, 5, 10)
EVALUATION_CUTOFFS = (10, 20)
MAXIMUM_CUTOFF = max(EVALUATION_CUTOFFS)
BATCH_SIZE = 128


def load_genre_features(
    file_path: Path,
) -> tuple[np.ndarray, sparse.csr_matrix]:
    """Load normalized track genre vectors as a sparse matrix."""

    columns = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        nrows=0,
    ).columns

    feature_dtypes = {
        column: "float32"
        for column in columns
        if column != "id"
    }

    frame = pd.read_csv(
        file_path,
        sep="\t",
        compression="bz2",
        dtype=feature_dtypes,
    )

    track_ids = frame.pop("id").astype(str).to_numpy()

    matrix = sparse.csr_matrix(
        frame.to_numpy(dtype=np.float32, copy=False)
    )

    matrix = normalize(
        matrix,
        norm="l2",
        axis=1,
        copy=False,
    ).tocsr()

    return track_ids, matrix


def attach_splits(
    interactions: pd.DataFrame,
    assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Attach user split assignments to interactions."""

    interactions = interactions.copy()
    assignments = assignments.copy()

    interactions["user"] = interactions["user"].astype(str)
    interactions["song"] = interactions["song"].astype(str)
    assignments["user"] = assignments["user"].astype(str)

    return interactions.merge(
        assignments,
        on="user",
        how="left",
        validate="many_to_one",
    )


def build_training_popularity(
    split_interactions: pd.DataFrame,
) -> pd.DataFrame:
    """Rank tracks using training-user interaction counts."""

    training = split_interactions[
        split_interactions["split"] == "train"
    ]

    popularity = (
        training.groupby("song", observed=True)
        .size()
        .rename("training_listeners")
        .reset_index()
        .sort_values(
            ["training_listeners", "song"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    popularity["popularity_rank"] = (
        np.arange(len(popularity)) + 1
    )

    return popularity


def build_exclusion_sets(
    profile: pd.DataFrame,
) -> dict[str, set[str]]:
    """Return all ten profile-pool tracks for each user."""

    prepared = profile.copy()
    prepared["user"] = prepared["user"].astype(str)
    prepared["song"] = prepared["song"].astype(str)

    return (
        prepared.groupby("user", observed=True)["song"]
        .apply(set)
        .to_dict()
    )


def recommend_popular_tracks(
    users: list[str],
    popularity_order: list[str],
    exclusion_sets: dict[str, set[str]],
    seed_size: int,
    cutoff: int,
) -> pd.DataFrame:
    """Recommend popular unseen tracks to each user."""

    rows = []

    for user in users:
        excluded = exclusion_sets[user]
        rank = 0

        for song in popularity_order:
            if song in excluded:
                continue

            rank += 1
            rows.append(
                {
                    "model": "popularity",
                    "user": user,
                    "seed_size": seed_size,
                    "rank": rank,
                    "song": song,
                }
            )

            if rank == cutoff:
                break

    return pd.DataFrame(rows)


def align_candidate_features(
    popularity: pd.DataFrame,
    genre_track_ids: np.ndarray,
    genre_matrix: sparse.csr_matrix,
) -> tuple[
    np.ndarray,
    sparse.csr_matrix,
    dict[str, int],
]:
    """Align the training catalog with the genre matrix."""

    genre_row_by_track = {
        track_id: row
        for row, track_id in enumerate(genre_track_ids)
    }

    candidate_tracks = popularity["song"].to_numpy()

    missing_tracks = [
        track
        for track in candidate_tracks
        if track not in genre_row_by_track
    ]

    if missing_tracks:
        raise ValueError(
            f"{len(missing_tracks)} training tracks lack genre rows."
        )

    genre_rows = [
        genre_row_by_track[track]
        for track in candidate_tracks
    ]

    candidate_matrix = genre_matrix[
        genre_rows
    ].tocsr()

    candidate_index = {
        track: index
        for index, track in enumerate(candidate_tracks)
    }

    return (
        candidate_tracks,
        candidate_matrix,
        candidate_index,
    )


def build_user_profiles(
    scenario_seeds: pd.DataFrame,
    users: list[str],
    genre_row_by_track: dict[str, int],
    genre_matrix: sparse.csr_matrix,
) -> sparse.csr_matrix:
    """Average and normalize each user's seed vectors."""

    seed_groups = {
        user: group["song"].astype(str).tolist()
        for user, group in scenario_seeds.groupby(
            "user",
            observed=True,
        )
    }

    profile_rows = []

    for user in users:
        genre_rows = [
            genre_row_by_track[song]
            for song in seed_groups[user]
            if song in genre_row_by_track
        ]

        if genre_rows:
            profile_vector = genre_matrix[
                genre_rows
            ].mean(axis=0)

            profile_rows.append(
                sparse.csr_matrix(
                    np.asarray(profile_vector)
                )
            )
        else:
            profile_rows.append(
                sparse.csr_matrix(
                    (1, genre_matrix.shape[1]),
                    dtype=np.float32,
                )
            )

    profiles = sparse.vstack(
        profile_rows,
        format="csr",
    )

    return normalize(
        profiles,
        norm="l2",
        axis=1,
        copy=False,
    ).tocsr()


def recommend_content_tracks(
    users: list[str],
    profiles: sparse.csr_matrix,
    candidate_tracks: np.ndarray,
    candidate_matrix: sparse.csr_matrix,
    candidate_index: dict[str, int],
    exclusion_sets: dict[str, set[str]],
    popularity_order: list[str],
    seed_size: int,
    cutoff: int,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Recommend tracks by cosine similarity to seed profiles."""

    rows = []
    candidate_transpose = candidate_matrix.transpose().tocsr()

    for batch_start in range(0, len(users), batch_size):
        batch_end = min(
            batch_start + batch_size,
            len(users),
        )

        batch_profiles = profiles[
            batch_start:batch_end
        ]

        batch_scores = (
            batch_profiles @ candidate_transpose
        ).toarray()

        for local_index, user_index in enumerate(
            range(batch_start, batch_end)
        ):
            user = users[user_index]
            scores = batch_scores[local_index]
            excluded = exclusion_sets[user]

            for song in excluded:
                index = candidate_index.get(song)

                if index is not None:
                    scores[index] = -np.inf

            if batch_profiles[local_index].nnz == 0:
                fallback = recommend_popular_tracks(
                    users=[user],
                    popularity_order=popularity_order,
                    exclusion_sets=exclusion_sets,
                    seed_size=seed_size,
                    cutoff=cutoff,
                )

                fallback["model"] = "content"
                rows.extend(fallback.to_dict("records"))
                continue

            top_indices = np.argpartition(
                scores,
                -cutoff,
            )[-cutoff:]

            order = np.lexsort(
                (
                    candidate_tracks[top_indices],
                    -scores[top_indices],
                )
            )

            top_indices = top_indices[order]

            for rank, candidate_position in enumerate(
                top_indices,
                start=1,
            ):
                rows.append(
                    {
                        "model": "content",
                        "user": user,
                        "seed_size": seed_size,
                        "rank": rank,
                        "song": candidate_tracks[
                            candidate_position
                        ],
                    }
                )

    return pd.DataFrame(rows)


def evaluate_recommendations(
    recommendations: pd.DataFrame,
    targets: pd.DataFrame,
    cutoffs: tuple[int, ...] = EVALUATION_CUTOFFS,
) -> pd.DataFrame:
    """Calculate mean Recall@K and NDCG@K."""

    target_sets = (
        targets.groupby("user", observed=True)["song"]
        .apply(set)
        .to_dict()
    )

    rows = []

    for (model, seed_size), group in recommendations.groupby(
        ["model", "seed_size"],
        observed=True,
    ):
        for cutoff in cutoffs:
            recalls = []
            ndcgs = []

            ideal_length = min(
                cutoff,
                len(next(iter(target_sets.values()))),
            )
            ideal_dcg = sum(
                1 / np.log2(rank + 1)
                for rank in range(1, ideal_length + 1)
            )

            for user, user_recommendations in group.groupby(
                "user",
                observed=True,
            ):
                ranked_tracks = (
                    user_recommendations.sort_values("rank")
                    .head(cutoff)["song"]
                    .tolist()
                )

                relevant_tracks = target_sets[user]
                relevance = [
                    int(track in relevant_tracks)
                    for track in ranked_tracks
                ]

                recalls.append(
                    sum(relevance) / len(relevant_tracks)
                )

                dcg = sum(
                    relevant / np.log2(rank + 1)
                    for rank, relevant in enumerate(
                        relevance,
                        start=1,
                    )
                )

                ndcgs.append(dcg / ideal_dcg)

            rows.append(
                {
                    "model": model,
                    "seed_size": seed_size,
                    "cutoff": cutoff,
                    "recall": np.mean(recalls),
                    "ndcg": np.mean(ndcgs),
                    "users": len(recalls),
                }
            )

    return pd.DataFrame(rows)


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Evaluate initial recommendation baselines."
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
        "--profile-path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
    )
    parser.add_argument(
        "--seeds-path",
        type=Path,
        default=DEFAULT_SEEDS_PATH,
    )
    parser.add_argument(
        "--targets-path",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
    )
    parser.add_argument(
        "--genres-path",
        type=Path,
        default=DEFAULT_GENRES_PATH,
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    return parser.parse_args()


def main() -> None:
    """Fit and evaluate popularity and content baselines."""

    args = parse_arguments()

    required_paths = [
        args.interactions_path,
        args.splits_path,
        args.profile_path,
        args.seeds_path,
        args.targets_path,
        args.genres_path,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading modeling tables...")
    interactions = pd.read_parquet(args.interactions_path)
    assignments = pd.read_parquet(args.splits_path)
    profile = pd.read_parquet(args.profile_path)
    seeds = pd.read_parquet(args.seeds_path)
    targets = pd.read_parquet(args.targets_path)

    profile = profile[profile["split"] == "validation"].copy()
    seeds = seeds[seeds["split"] == "validation"].copy()
    targets = targets[
        targets["split"] == "validation"
    ].copy()

    for frame in [profile, seeds, targets]:
        frame["user"] = frame["user"].astype(str)
        frame["song"] = frame["song"].astype(str)

    split_interactions = attach_splits(
        interactions,
        assignments,
    )

    popularity = build_training_popularity(
        split_interactions
    )
    popularity_order = popularity["song"].tolist()

    exclusion_sets = build_exclusion_sets(profile)
    users = sorted(profile["user"].unique())

    print("Loading sparse genre features...")
    genre_track_ids, genre_matrix = load_genre_features(
        args.genres_path
    )

    genre_row_by_track = {
        track_id: row
        for row, track_id in enumerate(genre_track_ids)
    }

    (
        candidate_tracks,
        candidate_matrix,
        candidate_index,
    ) = align_candidate_features(
        popularity,
        genre_track_ids,
        genre_matrix,
    )

    recommendation_frames = []

    for seed_size in SEED_SIZES:
        print(f"Evaluating seed size {seed_size}...")

        scenario_seeds = seeds[
            seeds["seed_size"] == seed_size
        ]

        popularity_recommendations = (
            recommend_popular_tracks(
                users=users,
                popularity_order=popularity_order,
                exclusion_sets=exclusion_sets,
                seed_size=seed_size,
                cutoff=MAXIMUM_CUTOFF,
            )
        )

        user_profiles = build_user_profiles(
            scenario_seeds=scenario_seeds,
            users=users,
            genre_row_by_track=genre_row_by_track,
            genre_matrix=genre_matrix,
        )

        content_recommendations = (
            recommend_content_tracks(
                users=users,
                profiles=user_profiles,
                candidate_tracks=candidate_tracks,
                candidate_matrix=candidate_matrix,
                candidate_index=candidate_index,
                exclusion_sets=exclusion_sets,
                popularity_order=popularity_order,
                seed_size=seed_size,
                cutoff=MAXIMUM_CUTOFF,
            )
        )

        recommendation_frames.extend(
            [
                popularity_recommendations,
                content_recommendations,
            ]
        )

    recommendations = pd.concat(
        recommendation_frames,
        ignore_index=True,
    )

    metrics = evaluate_recommendations(
        recommendations,
        targets,
    ).sort_values(
        ["cutoff", "seed_size", "model"]
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_parquet(
        args.output_directory
        / "validation_baseline_recommendations.parquet",
        index=False,
    )

    metrics.to_csv(
        args.output_directory
        / "validation_baseline_metrics.csv",
        index=False,
    )

    print("\nValidation baseline metrics")
    print(metrics.to_string(index=False))

    print(
        "\nSaved baseline results to:",
        args.output_directory,
    )


if __name__ == "__main__":
    main()