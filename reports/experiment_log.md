# Experiment Log

## Validation Design

Models are evaluated on 1,399 unseen validation users. For each user:

- the first 10 distinct tracks form a fixed profile pool;
- 1, 3, 5, or 10 profile tracks are revealed as seeds;
- the next 10 distinct tracks form an identical target set across scenarios;
- all profile-pool tracks are excluded from recommendations;
- model statistics are learned only from training users.

The test split remains untouched during model development.

## Initial Baselines

| Model | Seeds | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|---:|
| Popularity | 1 | 0.0333 | 0.0397 | 0.0490 | 0.0482 |
| Content | 1 | 0.0031 | 0.0032 | 0.0054 | 0.0045 |
| Content | 3 | 0.0031 | 0.0035 | 0.0049 | 0.0045 |
| Content | 5 | 0.0044 | 0.0041 | 0.0071 | 0.0057 |
| Content | 10 | 0.0070 | 0.0085 | 0.0109 | 0.0107 |

Popularity is independent of seed size because it produces the same global ranking for every scenario, apart from excluding the fixed profile pool.

## Findings

The popularity baseline substantially outperforms genre-content similarity. This is consistent with the dataset’s strong popularity concentration: approximately 34% of listening events involve the top 1% of tracks.

Content performance generally improves as more seed tracks become available, but the improvement is not monotonic at every cutoff. Averaging multiple genre vectors may dilute distinct or inconsistent musical preferences, and genre features alone may be too coarse to predict exact next-track choices.

These findings establish two requirements for subsequent models:

1. Personalized models must outperform the popularity baseline rather than merely outperforming random recommendation.
2. Hybridization should account for the quantity and coherence of seed evidence instead of relying only on a fixed average content profile.

All results above are validation results. Test users will remain untouched until model selection is complete.

## Collaborative Latent-Factor Baseline

A truncated-SVD collaborative model was trained on the binary interaction matrix containing 11,190 training users, 67,428 tracks, and 2,003,815 nonzero interactions.

Latent dimensionality was selected using validation performance.

| Components | Explained variance | Recall@10, 1 seed | Recall@10, 3 seeds | Recall@10, 5 seeds | Recall@10, 10 seeds |
|---:|---:|---:|---:|---:|---:|
| 32 | 10.08% | 0.0305 | 0.0347 | 0.0423 | 0.0585 |
| 64 | 13.66% | 0.0325 | 0.0385 | 0.0461 | 0.0713 |
| 128 | 18.45% | 0.0341 | 0.0434 | 0.0520 | 0.0822 |
| 256 | 24.77% | **0.0352** | **0.0469** | **0.0565** | **0.0924** |

The 256-component model was selected because it achieved the strongest ranking performance at every cold-start level.

Collaborative performance improves substantially as more seeds become available. With one seed, the model only narrowly exceeds popularity. With 10 seeds, it improves Recall@10 by approximately 177% and NDCG@10 by approximately 172% over popularity.

This supports the central cold-start hypothesis: collaborative evidence becomes much more useful as the system observes additional user preferences.