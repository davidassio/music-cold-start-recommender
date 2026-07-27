# Problem Statement

## Objective

Build a hybrid music recommendation system that generates relevant,
novel, and diverse song recommendations for users with little or no
historical interaction data.

## Prediction Task

Given a user's limited set of seed interactions, rank unseen songs by
the likelihood that the user will engage with them.

## Cold-Start Scenarios

The system will be evaluated with 1, 3, and 5 observed seed songs per
new user. Performance will also be measured for warm users with longer
interaction histories.

## Modeling Strategy

The project will compare:

1. Popularity-based recommendations
2. Content-based recommendations
3. Collaborative filtering
4. Hybrid recommendation
5. Neural two-tower retrieval

The hybrid system will adjust the balance between content and
collaborative signals according to the amount of information available
about the user.

## Evaluation

Primary ranking metrics:

- Recall@K
- NDCG@K

Beyond-accuracy metrics:

- Catalog coverage
- Novelty
- Intra-list diversity

## Dataset

The primary dataset is Music4All-Onion, which combines anonymized
user-song listening records with song-level genre, tag, lyrics, and
audio-derived features.