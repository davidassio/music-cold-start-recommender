# Music Cold-Start Recommender

A hybrid music recommendation system designed to generate relevant, novel, and diverse recommendations for users with limited listening history.

## Overview

Traditional collaborative filtering relies on historical user behavior. It therefore struggles when a new listener has provided few or no interactions—a problem known as **user cold start**.

This project investigates how content features and collaborative signals can be combined to improve recommendations during the early stages of a user’s experience.

Given a small set of seed songs, the system will rank unseen tracks that the listener is likely to engage with.

## Cold-Start Scenarios

The recommendation system will be evaluated under several levels of available user history:

| Scenario            | Observed seed songs |
| ------------------- | ------------------: |
| Severe cold start   |                   1 |
| Moderate cold start |                   3 |
| Light cold start    |                   5 |
| Warm user           |          10 or more |

The hybrid model will place greater weight on content features when behavioral data is limited and progressively incorporate collaborative signals as more interactions become available.

## Dataset

This project uses the [Music4All-Onion dataset](https://zenodo.org/records/6609677), which combines anonymized listening behavior with song-level content information.

The current pipeline uses:

* `userid_trackid_count.tsv.bz2`: aggregated user–track play counts
* `id_genres_tf-idf.tsv.bz2`: TF-IDF genre representations

Raw dataset files are excluded from Git because of their size. Reproducible download and integrity-checking instructions are provided below.

## Dataset Audit Results

The interaction data was audited directly from the compressed source file using chunked processing.

| Statistic          |       Value |
| ------------------ | ----------: |
| User–track pairs   |  50,016,042 |
| Unique users       |     119,140 |
| Unique tracks      |      56,512 |
| Total plays        | 252,984,396 |
| Minimum play count |           1 |
| Maximum play count |      60,863 |
| Matrix density     |   0.742867% |
| Matrix sparsity    |  99.257133% |

The interaction matrix contains approximately 6.73 billion possible user–track combinations. Because more than 99% of these combinations are unobserved, later modeling stages will use sparse data structures rather than a dense matrix.

## Modeling Roadmap

The project will compare progressively more sophisticated approaches:

1. Popularity baseline
2. Content-based recommendation
3. Implicit-feedback collaborative filtering
4. Hybrid recommendation
5. Neural two-tower retrieval model
6. Diversity- and novelty-aware reranking
7. Interactive recommendation demo

Each advanced model will be compared against simpler baselines to determine whether its additional complexity produces a meaningful improvement.

## Evaluation

The primary ranking metrics will be:

* **Recall@K**: the proportion of relevant held-out tracks recovered
* **NDCG@K**: the quality of the recommendation ordering

The project will also measure:

* Catalog coverage
* Recommendation novelty
* Intra-list diversity

Cold-start evaluation will reveal only the first 1, 3, or 5 seed interactions for a held-out user. Recommendations will then be evaluated against that user’s later interactions.

## Repository Structure

```text
music-cold-start-recommender/
├── configs/                    # Experiment and model configuration
├── data/
│   ├── raw/                    # Immutable source data
│   ├── interim/                # Partially transformed data
│   └── processed/              # Model-ready datasets
├── docs/
│   └── problem_statement.md    # Formal problem definition
├── models/                     # Serialized model artifacts
├── notebooks/                  # Exploration and experiments
├── reports/
│   └── figures/                # Final visualizations
├── src/
│   └── music_recommender/
│       ├── __init__.py
│       └── data_audit.py       # Memory-efficient dataset audit
├── tests/                      # Automated tests
├── pyproject.toml
└── README.md
```

## Installation

This project uses Python 3.12.

Clone the repository and enter its directory:

```bash
git clone https://github.com/davidassio/music-cold-start-recommender.git
cd music-cold-start-recommender
```

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the project in editable mode:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Data Acquisition

Download the genre features:

```bash
curl -L --continue-at - \
  --output data/raw/id_genres_tf-idf.tsv.bz2 \
  "https://zenodo.org/records/6609677/files/id_genres_tf-idf.tsv.bz2?download=1"
```

Download the aggregated interaction data:

```bash
curl -L --continue-at - \
  --output data/raw/userid_trackid_count.tsv.bz2 \
  "https://zenodo.org/records/6609677/files/userid_trackid_count.tsv.bz2?download=1"
```

Verify the downloads on macOS:

```bash
md5 data/raw/id_genres_tf-idf.tsv.bz2
md5 data/raw/userid_trackid_count.tsv.bz2
```

Expected MD5 checksums:

| File                           | MD5 checksum                       |
| ------------------------------ | ---------------------------------- |
| `id_genres_tf-idf.tsv.bz2`     | `a742b5fa1d2e2ce780101773e57bb7f5` |
| `userid_trackid_count.tsv.bz2` | `314b51196a9c8f333c7fefc0711760a1` |

## Running the Data Audit

Run a quick validation over two chunks:

```bash
python -m music_recommender.data_audit --max-chunks 2
```

Run the complete interaction audit:

```bash
python -m music_recommender.data_audit
```

The audit reads the compressed dataset in 250,000-row chunks. This keeps memory usage bounded while calculating statistics across approximately 50 million user–track pairs.

An alternative chunk size can be provided:

```bash
python -m music_recommender.data_audit --chunk-size 500000
```

## Current Status

* [x] Repository and Python environment initialized
* [x] Cold-start problem formally defined
* [x] Raw data downloaded and checksum-verified
* [x] Memory-efficient interaction audit implemented
* [x] Complete interaction dataset audited
* [ ] Genre features audited and joined
* [ ] Modeling subset constructed
* [ ] Popularity baseline implemented
* [ ] Content-based recommender implemented
* [ ] Collaborative filtering implemented
* [ ] Hybrid recommender implemented
* [ ] Neural retrieval model implemented
* [ ] Interactive demo created

## Data Attribution

Music4All-Onion is provided by its original authors through Zenodo. Dataset files are not redistributed through this repository. Users should consult the official dataset page for its documentation, citation requirements, and licensing terms.
