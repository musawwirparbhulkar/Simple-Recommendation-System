# Simple Movie Recommender

A small, dependency-free content-based recommender. You give it a movie
title or a list of tags; it returns the most similar films from a
hand-curated catalog, with an explanation of *why* each was suggested.

## How it works

1. **Catalog with tags.** Every movie in `MOVIES` is described by a
   short list of tags covering genre, mood, theme, and style
   (e.g. *Inception* → `sci-fi, action, mind-bending, heist, thriller`).

2. **Multi-hot encoding.** All unique tags across the catalog form a
   fixed vocabulary. Each movie becomes a binary vector of length
   `|vocab|`, with a `1` for every tag it carries and `0` elsewhere.

3. **Preference vector.** The user's input is turned into the same
   kind of vector:
   - If they name a movie, we copy that movie's vector.
   - If they list keywords (e.g. `sci-fi, mind-bending, emotional`),
     we set the corresponding vocabulary positions to `1`.

4. **Cosine similarity.** For every movie we compute
   `cos(preference, movie) = dot / (||preference|| * ||movie||)`.
   Cosine similarity is a natural fit here because it ignores the
   absolute number of shared tags and measures the *angle* between
   the two tag profiles — so a small movie and a large movie that
   share the same flavor still score well.

5. **Rank & explain.** We sort by score, drop the seed movie (if any)
   from the result list, and report the top N. For each pick we also
   list the *shared tags* with the preference so the user can see
   the reasoning — no black-box "you might also like" hand-waving.

## Why cosine over Jaccard?

Jaccard (intersection / union) treats every tag as equally rare, but
cosine over binary vectors already gives a very similar ranking on
this kind of data, scales naturally if the catalog grows, and is the
more familiar formulation in the recommender-systems literature. For
a dataset this small the two metrics would rank almost identically.

## Run it

```bash
python movie_recommender.py            # interactive CLI
python movie_recommender.py --selftest # a few canned queries
```

### Interactive example

```
> Inception

Because you liked 'Inception', you might also enjoy:
----------------------------------------
  1. The Matrix (1999)  similarity=0.632
     why: shared tags -> sci-fi, action, mind-bending
  2. Everything Everywhere All at Once (2022)  similarity=0.577
     why: shared tags -> sci-fi, mind-bending
  ...
```

## Files

- `movie_recommender.py` — the whole engine + CLI.
- `README.md` — this file.

## What you'd add for a real system

- A real ID-based catalog (TMDB / Goodreads) instead of hard-coded
  tags.
- TF-IDF weighting so generic tags (`drama`) count less than specific
  ones (`cyberpunk`).
- Embedding-based similarity (e.g. sentence transformers on plot
  summaries) once you outgrow bag-of-tags.
- Collaborative filtering once you have interaction logs.
