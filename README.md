# Music Cold-Start Recommender

A hybrid music recommendation system designed to generate relevant, novel, and diverse recommendations for listeners with limited history.

## Overview

Traditional collaborative filtering depends on historical user behavior. It therefore struggles when a new listener has provided few interactions—a problem known as **user cold start**.

This project investigates how content features and collaborative signals can be combined during the early stages of a user’s experience. Given a small set of seed songs, the system ranks unseen tracks that the listener may engage with.

## Cold-Start Scenarios

The system will be evaluated under several levels of observed user history:

| Scenario            | Observed seed songs |
| ------------------- | ------------------: |
| Severe cold start   |                   1 |
| Moderate cold start |                   3 |
| Light cold start    |                   5 |
| Warm user           |                  10 |

The hybrid model will initially rely heavily on content features and progressively incorporate collaborative signals as more interactions become available.

## Datasets

The project uses two related datasets.

### Music4All-Onion

[Music4All-Onion](https://zenodo.org/records/6609677) supplies large-scale aggregated listening behavior and sparse genre representations:

* `userid_trackid_count.tsv.bz2`: aggregated user–track play counts
* `id_genres_tf-idf.tsv.bz2`: 685-dimensional genre TF-IDF vectors

This dataset supports large-scale implicit-feedback and collaborative-filtering experiments.

### Music4All

The original Music4All dataset supplies:

* Timestamped listening events
* Artist, song, and album names
* Spotify identifiers and audio attributes
* Genres and descriptive tags
* Lyric-language labels

The project uses the following tabular files:

* `listening_history.csv`
* `id_information.csv`
* `id_metadata.csv`
* `id_genres.csv`
* `id_tags.csv`
* `id_lang.csv`

These files were obtained directly from the dataset authors under a disclosure agreement. They are excluded from Git and are not redistributed by this repository.

The timestamped listening history enables chronological cold-start evaluation, while the readable metadata allows recommendations and visualizations to use actual artist and song names.

## Dataset Audit Results

### Aggregated Music4All-Onion Interactions

The complete interaction file was audited with memory-efficient chunked processing.

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

The interaction matrix contains approximately 6.73 billion possible user–track combinations. Because more than 99% are unobserved, modeling stages will use sparse data structures.

All 56,512 interaction tracks have corresponding genre records.

### Genre TF-IDF Features

| Statistic                    |      Value |
| ---------------------------- | ---------: |
| Tracks                       |    109,269 |
| Genre features               |        685 |
| Total feature cells          | 74,849,260 |
| Nonzero feature cells        |    237,448 |
| Feature density              |  0.317235% |
| Feature sparsity             | 99.682765% |
| Tracks without active genres |         27 |

The content matrix is extremely sparse, with approximately 2.17 active genre features per track on average.

### Music4All Metadata

Each of the five track-level metadata tables contains:

* 109,269 unique track IDs
* No duplicate track IDs
* No missing track IDs
* No missing cells
* Complete one-to-one coverage across tables

This provides artist, title, album, Spotify, genre, tag, language, and audio-feature information for the full catalog.

### Timestamped Listening History

| Statistic                |      Value |
| ------------------------ | ---------: |
| Listening events         |  5,109,592 |
| Unique users             |     14,127 |
| Unique tracks            |     99,596 |
| Metadata coverage        |       100% |
| Earliest timestamp       | 2013-12-30 |
| Latest timestamp         | 2019-03-26 |
| Median events per user   |        375 |
| Mean events per user     |     361.69 |
| Maximum events per user  |        500 |
| Median events per track  |         11 |
| Maximum events per track |     82,871 |

The exact maximum of 500 events per user suggests that listening histories were capped during dataset construction. This limitation will be considered during EDA and evaluation design.

## Evaluation Design

The primary evaluation will use the timestamped Music4All listening history.

For each eligible user:

1. Interactions will be ordered chronologically.
2. Only the first 1, 3, 5, or 10 distinct seed tracks will be revealed.
3. The recommender will rank unseen candidate tracks.
4. Recommendations will be evaluated against the user’s later interactions.

This simulates how the system would perform as a new listener gradually provides more behavioral evidence.

The primary ranking metrics will be:

* **Recall@K:** proportion of relevant held-out tracks recovered
* **NDCG@K:** ranking quality with greater credit for relevant tracks near the top

The project will also measure:

* Catalog coverage
* Recommendation novelty
* Intra-list diversity

Music4All-Onion will support secondary large-scale implicit-feedback experiments and comparisons.

## Modeling Roadmap

The project will compare progressively more sophisticated approaches:

1. Popularity baseline
2. Content-based recommendation
3. Implicit-feedback collaborative filtering
4. Hybrid recommendation
5. Neural two-tower retrieval model
6. Diversity- and novelty-aware reranking
7. Interactive recommendation demo

Each advanced approach will be compared with simpler baselines to determine whether its additional complexity produces a meaningful improvement.

## Repository Structure

```text
music-cold-start-recommender/
├── configs/                         # Experiment and model configuration
├── data/
│   ├── raw/                         # Immutable, Git-ignored source data
│   ├── interim/                     # Partially transformed data
│   └── processed/                   # Model-ready datasets
├── docs/
│   └── problem_statement.md         # Formal problem definition
├── models/                          # Serialized model artifacts
├── notebooks/
│   └── 01_data_eda.ipynb            # Exploratory data analysis
├── reports/
│   ├── figures/                     # Final visualizations
│   └── tables/                      # Generated audit and EDA tables
├── src/
│   └── music_recommender/
│       ├── __init__.py
│       ├── data_audit.py            # Aggregated interaction audit
│       ├── genre_audit.py           # Genre TF-IDF audit
│       ├── coverage_audit.py        # Interaction/content coverage
│       ├── metadata_audit.py        # Protected metadata validation
│       └── listening_history_audit.py
├── tests/                           # Automated tests
├── pyproject.toml
└── README.md
```

## Installation

This project uses Python 3.12.

Clone the repository:

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

## Public Data Acquisition

Download the Music4All-Onion genre features:

```bash
curl -L --continue-at - \
  --output data/raw/id_genres_tf-idf.tsv.bz2 \
  "https://zenodo.org/records/6609677/files/id_genres_tf-idf.tsv.bz2?download=1"
```

Download the aggregated interactions:

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

Expected checksums:

| File                           | MD5 checksum                       |
| ------------------------------ | ---------------------------------- |
| `id_genres_tf-idf.tsv.bz2`     | `a742b5fa1d2e2ce780101773e57bb7f5` |
| `userid_trackid_count.tsv.bz2` | `314b51196a9c8f333c7fefc0711760a1` |

Access to the original Music4All files must be requested from the dataset authors. Those files cannot be downloaded or redistributed through this repository.

## Running the Audits

Run the aggregated interaction audit:

```bash
python -m music_recommender.data_audit
```

Run the genre-feature audit:

```bash
python -m music_recommender.genre_audit
```

Measure interaction and genre coverage:

```bash
python -m music_recommender.coverage_audit
```

If the agreement-protected Music4All files are available locally, run:

```bash
python -m music_recommender.metadata_audit
python -m music_recommender.listening_history_audit
```

The large interaction files are processed in chunks to keep memory usage bounded.

## Current Status

* [x] Repository and Python environment initialized
* [x] Cold-start problem formally defined
* [x] Public Music4All-Onion data downloaded and verified
* [x] Aggregated interaction dataset audited
* [x] Genre TF-IDF features audited
* [x] Interaction and genre coverage validated
* [x] Music4All metadata obtained and audited locally
* [x] Timestamped listening history audited
* [ ] Interaction and content EDA completed
* [ ] Evidence-based filtering thresholds selected
* [ ] Chronological train/validation/test splits created
* [ ] Popularity baseline implemented
* [ ] Content-based recommender implemented
* [ ] Collaborative filtering implemented
* [ ] Hybrid recommender implemented
* [ ] Neural retrieval model implemented
* [ ] Interactive demo created

## Data Attribution

Music4All-Onion is provided by its authors through Zenodo. The original Music4All dataset was obtained separately from its authors under a disclosure agreement.

Dataset files are not redistributed through this repository. Users should consult the official dataset documentation for licensing, citation, and access requirements.
