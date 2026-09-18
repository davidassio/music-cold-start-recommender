"""Fit and evaluate a collaborative latent-factor baseline."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from music_recommender.baselines import (
    EVALUATION_CUTOFFS,
    MAXIMUM_CUTOFF,
    SEED_SIZES,
    attach_splits,
    build_exclusion_sets,
    build_training_popularity,
    evaluate_recommendations,
    recommend_popular_tracks,
)

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
DEFAULT_OUTPUT_DIRECTORY = Path(
    "data/interim/results"
)

LATENT_COMPONENTS = 64
RANDOM_SEED = 42
BATCH_SIZE = 128


def build_training_matrix(
    split_interactions: pd.DataFrame,
    candidate_tracks: np.ndarray,
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    """Build a binary training-user-by-track matrix."""

    training = split_interactions[
        split_interactions["split"] == "train"
    ].copy()

    training["user"] = training["user"].astype(str)
    training["song"] = training["song"].astype(str)

    training_users = np.sort(training["user"].unique())

    user_index = {
        user: index
        for index, user in enumerate(training_users)
    }
    track_index = {
        track: index
        for index, track in enumerate(candidate_tracks)
    }

    row_indices = (
        training["user"].map(user_index).to_numpy()
    )
    column_indices = (
        training["song"].map(track_index).to_numpy()
    )

    matrix = sparse.csr_matrix(
        (
            np.ones(len(training), dtype=np.float32),
            (row_indices, column_indices),
        ),
        shape=(
            len(training_users),
            len(candidate_tracks),
        ),
        dtype=np.float32,
    )

    return matrix, track_index


def fit_item_embeddings(
    training_matrix: sparse.csr_matrix,
    component_count: int = LATENT_COMPONENTS,
) -> tuple[np.ndarray, TruncatedSVD]:
    """Learn normalized collaborative track embeddings."""

    model = TruncatedSVD(
        n_components=component_count,
        n_iter=7,
        random_state=RANDOM_SEED,
    )

    model.fit(training_matrix)

    item_embeddings = (
        model.components_.T
        * model.singular_values_
    ).astype(np.float32)

    item_embeddings = normalize(
        item_embeddings,
        norm="l2",
        axis=1,
        copy=False,
    )

    return item_embeddings, model


def build_collaborative_profiles(
    scenario_seeds: pd.DataFrame,
    users: list[str],
    track_index: dict[str, int],
    item_embeddings: np.ndarray,
) -> np.ndarray:
    """Average each validation user's seed embeddings."""

    seed_groups = {
        user: group["song"].astype(str).tolist()
        for user, group in scenario_seeds.groupby(
            "user",
            observed=True,
        )
    }

    profiles = np.zeros(
        (len(users), item_embeddings.shape[1]),
        dtype=np.float32,
    )

    for user_position, user in enumerate(users):
        seed_indices = [
            track_index[song]
            for song in seed_groups[user]
            if song in track_index
        ]

        if seed_indices:
            profiles[user_position] = item_embeddings[
                seed_indices
            ].mean(axis=0)

    return normalize(
        profiles,
        norm="l2",
        axis=1,
        copy=False,
    )


def recommend_collaborative_tracks(
    users: list[str],
    profiles: np.ndarray,
    candidate_tracks: np.ndarray,
    item_embeddings: np.ndarray,
    track_index: dict[str, int],
    exclusion_sets: dict[str, set[str]],
    popularity_order: list[str],
    seed_size: int,
    cutoff: int,
    batch_size: int = BATCH_SIZE,
) -> pd.DataFrame:
    """Rank candidates by latent-factor cosine similarity."""

    rows = []

    for batch_start in range(0, len(users), batch_size):
        batch_end = min(
            batch_start + batch_size,
            len(users),
        )

        batch_profiles = profiles[
            batch_start:batch_end
        ]
        batch_scores = (
            batch_profiles @ item_embeddings.T
        )

        for local_index, user_position in enumerate(
            range(batch_start, batch_end)
        ):
            user = users[user_position]
            scores = batch_scores[local_index]
            excluded = exclusion_sets[user]

            if not np.any(batch_profiles[local_index]):
                fallback = recommend_popular_tracks(
                    users=[user],
                    popularity_order=popularity_order,
                    exclusion_sets=exclusion_sets,
                    seed_size=seed_size,
                    cutoff=cutoff,
                )

                fallback["model"] = "collaborative"
                rows.extend(fallback.to_dict("records"))
                continue

            for song in excluded:
                candidate_position = track_index.get(song)

                if candidate_position is not None:
                    scores[candidate_position] = -np.inf

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
                        "model": "collaborative",
                        "user": user,
                        "seed_size": seed_size,
                        "rank": rank,
                        "song": candidate_tracks[
                            candidate_position
                        ],
                    }
                )

    return pd.DataFrame(rows)


def parse_arguments():
    """Read command-line arguments."""

    parser = ArgumentParser(
        description="Evaluate a collaborative SVD baseline."
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
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--components",
        type=int,
        default=LATENT_COMPONENTS,
    )

    return parser.parse_args()


def main() -> None:
    """Fit and evaluate collaborative recommendations."""

    args = parse_arguments()

    for path in [
        args.interactions_path,
        args.splits_path,
        args.profile_path,
        args.seeds_path,
        args.targets_path,
    ]:
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
    candidate_tracks = popularity["song"].to_numpy()
    popularity_order = popularity["song"].tolist()

    print("Building sparse training matrix...")
    training_matrix, track_index = build_training_matrix(
        split_interactions,
        candidate_tracks,
    )

    print(
        "Training matrix:",
        f"{training_matrix.shape[0]:,} users x",
        f"{training_matrix.shape[1]:,} tracks",
    )
    print(
        "Nonzero interactions:",
        f"{training_matrix.nnz:,}",
    )

    print(
        f"Fitting {args.components}-component SVD..."
    )
    item_embeddings, model = fit_item_embeddings(
        training_matrix,
        component_count=args.components,
    )

    print(
        "Explained variance:",
        f"{model.explained_variance_ratio_.sum():.2%}",
    )

    users = sorted(profile["user"].unique())
    exclusion_sets = build_exclusion_sets(profile)
    recommendation_frames = []

    for seed_size in SEED_SIZES:
        print(f"Evaluating seed size {seed_size}...")

        scenario_seeds = seeds[
            seeds["seed_size"] == seed_size
        ]

        user_profiles = build_collaborative_profiles(
            scenario_seeds=scenario_seeds,
            users=users,
            track_index=track_index,
            item_embeddings=item_embeddings,
        )

        recommendations = (
            recommend_collaborative_tracks(
                users=users,
                profiles=user_profiles,
                candidate_tracks=candidate_tracks,
                item_embeddings=item_embeddings,
                track_index=track_index,
                exclusion_sets=exclusion_sets,
                popularity_order=popularity_order,
                seed_size=seed_size,
                cutoff=MAXIMUM_CUTOFF,
            )
        )

        recommendation_frames.append(recommendations)

    recommendations = pd.concat(
        recommendation_frames,
        ignore_index=True,
    )

    metrics = evaluate_recommendations(
        recommendations,
        targets,
        cutoffs=EVALUATION_CUTOFFS,
    ).sort_values(
        ["cutoff", "seed_size"]
    )

    args.output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendations.to_parquet(
        args.output_directory
        / "validation_collaborative_recommendations.parquet",
        index=False,
    )
    metrics.to_csv(
        args.output_directory
        / "validation_collaborative_metrics_{args.componenets}.csv",
        index=False,
    )

    print("\nValidation collaborative metrics")
    print(metrics.to_string(index=False))

    print(
        "\nSaved collaborative results to:",
        args.output_directory,
    )


if __name__ == "__main__":
    main()