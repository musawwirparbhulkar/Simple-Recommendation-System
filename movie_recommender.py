"""
Simple Movie Recommendation System
====================================
A content-based recommender that suggests movies similar to a user's
preference using cosine similarity over multi-hot feature vectors.

How it works (the short version):
  1. Each movie in the catalog is described by a set of tags drawn from
     four buckets: genre, mood, theme, and style.
  2. We build a vocabulary of all unique tags, then turn every movie
     into a binary (multi-hot) vector: 1 if the tag is present, else 0.
  3. The user's preference is converted into the same kind of vector:
        - if they name a movie, we use that movie's vector, or
        - if they list keywords, we mark those tags as 1 directly.
  4. We compute cosine similarity between the preference vector and
     every movie vector. The top N (excluding the seed movie itself)
     are returned, each with its score and the tags they share with
     the preference, so the user can see *why* it was suggested.

No external libraries are required - everything runs on the Python
standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


# ---------------------------------------------------------------------------
# 1. The catalog
# ---------------------------------------------------------------------------
# A small, hand-picked set of films. Tags are intentionally simple so the
# similarity logic is easy to follow. Each tag should be a single token
# (lowercase, hyphenated) so multi-hot encoding works cleanly.
MOVIES: list[dict] = [
    {
        "title": "The Matrix",
        "year": 1999,
        "tags": ["sci-fi", "action", "mind-bending", "dystopian", "cyberpunk"],
    },
    {
        "title": "Inception",
        "year": 2010,
        "tags": ["sci-fi", "action", "mind-bending", "heist", "thriller"],
    },
    {
        "title": "Interstellar",
        "year": 2014,
        "tags": ["sci-fi", "drama", "emotional", "space", "epic"],
    },
    {
        "title": "The Dark Knight",
        "year": 2008,
        "tags": ["action", "crime", "thriller", "dark", "superhero"],
    },
    {
        "title": "Pulp Fiction",
        "year": 1994,
        "tags": ["crime", "dark-comedy", "thriller", "non-linear", "dialogue-driven"],
    },
    {
        "title": "Forrest Gump",
        "year": 1994,
        "tags": ["drama", "feel-good", "emotional", "romance", "historical"],
    },
    {
        "title": "The Shawshank Redemption",
        "year": 1994,
        "tags": ["drama", "emotional", "hope", "prison", "friendship"],
    },
    {
        "title": "Spirited Away",
        "year": 2001,
        "tags": ["animation", "fantasy", "magical", "family", "adventure"],
    },
    {
        "title": "Parasite",
        "year": 2019,
        "tags": ["thriller", "dark-comedy", "social-commentary", "twist", "class"],
    },
    {
        "title": "Whiplash",
        "year": 2014,
        "tags": ["drama", "intense", "music", "obsession", "character-study"],
    },
    {
        "title": "La La Land",
        "year": 2016,
        "tags": ["romance", "musical", "bittersweet", "dreamy", "performances"],
    },
    {
        "title": "The Grand Budapest Hotel",
        "year": 2014,
        "tags": ["comedy", "quirky", "visual", "whimsical", "adventure"],
    },
    {
        "title": "Mad Max: Fury Road",
        "year": 2015,
        "tags": ["action", "post-apocalyptic", "high-octane", "visual", "intense"],
    },
    {
        "title": "Her",
        "year": 2013,
        "tags": ["romance", "sci-fi", "emotional", "intimate", "thought-provoking"],
    },
    {
        "title": "Everything Everywhere All at Once",
        "year": 2022,
        "tags": ["sci-fi", "comedy", "emotional", "mind-bending", "family", "surreal"],
    },
    {
        "title": "Knives Out",
        "year": 2019,
        "tags": ["mystery", "comedy", "twist", "whodunit", "dialogue-driven"],
    },
    {
        "title": "Coco",
        "year": 2017,
        "tags": ["animation", "family", "emotional", "music", "feel-good", "cultural"],
    },
]


# ---------------------------------------------------------------------------
# 2. Vector encoding
# ---------------------------------------------------------------------------
def build_vocab(movies: Iterable[dict]) -> list[str]:
    """Collect every unique tag across the catalog, sorted for stability."""
    seen: set[str] = set()
    for m in movies:
        seen.update(m["tags"])
    return sorted(seen)


def movie_to_vector(movie: dict, vocab: list[str]) -> list[int]:
    """Turn a movie's tag list into a multi-hot vector over `vocab`."""
    tagset = set(movie["tags"])
    return [1 if tag in tagset else 0 for tag in vocab]


def tags_to_vector(tags: Iterable[str], vocab: list[str]) -> list[int]:
    """Turn a free-form list of user-entered tags into the same vector form."""
    tagset = {normalize_tag(t) for t in tags if t.strip()}
    return [1 if tag in tagset else 0 for tag in vocab]


def normalize_tag(raw: str) -> str:
    """Lowercase, strip whitespace, and replace spaces with hyphens so user
    input matches the catalog tag convention ('dark comedy' -> 'dark-comedy')."""
    return re.sub(r"\s+", "-", raw.strip().lower())


# ---------------------------------------------------------------------------
# 3. Similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[int], b: list[int]) -> float:
    """Standard cosine similarity between two equally-sized binary vectors.
    Returns 0.0 if either vector is all zeros (no shared signal to compare)."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def shared_tags(a: list[int], b: list[int], vocab: list[str]) -> list[str]:
    """Return the tags that are 'on' in both vectors - useful for explaining
    *why* a recommendation was made."""
    return [vocab[i] for i, (x, y) in enumerate(zip(a, b)) if x == 1 and y == 1]


# ---------------------------------------------------------------------------
# 4. Recommendation engine
# ---------------------------------------------------------------------------
def build_movie_index(movies: list[dict]) -> dict[str, dict]:
    """Map title (case-insensitive) -> movie dict, so user lookups are easy."""
    return {m["title"].lower(): m for m in movies}


def recommend(
    preference_vector: list[int],
    vocab: list[str],
    movies: list[dict],
    vectors: list[list[int]],
    exclude_title: str | None = None,
    top_n: int = 5,
) -> list[dict]:
    """Score every movie against the preference and return the top N.
    If `exclude_title` is provided (e.g. the seed movie the user already
    liked), that movie is removed from the results."""
    scored: list[dict] = []
    for movie, vec in zip(movies, vectors):
        if exclude_title and movie["title"].lower() == exclude_title.lower():
            continue
        score = cosine_similarity(preference_vector, vec)
        scored.append(
            {
                "title": movie["title"],
                "year": movie["year"],
                "score": score,
                "shared": shared_tags(preference_vector, vec, vocab),
            }
        )
    scored.sort(key=lambda r: r["score"], reverse=True)
    return scored[:top_n]


# ---------------------------------------------------------------------------
# 5. Pretty printing
# ---------------------------------------------------------------------------
def print_recommendations(results: list[dict], header: str) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    if not results or results[0]["score"] == 0.0:
        print("  (no strong matches found - try different keywords)")
        return
    for rank, r in enumerate(results, start=1):
        shared = ", ".join(r["shared"]) if r["shared"] else "no tag overlap"
        print(
            f"  {rank}. {r['title']} ({r['year']})  "
            f"similarity={r['score']:.3f}"
        )
        print(f"     why: shared tags -> {shared}")


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
BANNER = """
+------------------------------------------+
|   Simple Movie Recommender (content-     |
|   based, cosine similarity, stdlib only) |
+------------------------------------------+
"""


HELP_TEXT = """
How to use
----------
1) Tell me what you liked. You can either:
   a) Type a movie title from the catalog (e.g. "Inception")
   b) Type a list of tags you enjoy, comma-separated
      (e.g. "sci-fi, mind-bending, emotional")

2) I'll show the top 5 most similar movies and why.

Commands:
   list         show every movie in the catalog
   tags <name>  show the tags for a given movie
   help         show this message
   quit         exit
"""


def interactive_loop() -> None:
    print(BANNER)
    movies = MOVIES
    vocab = build_vocab(movies)
    vectors = [movie_to_vector(m, vocab) for m in movies]
    index = build_movie_index(movies)

    print(f"Loaded {len(movies)} movies, {len(vocab)} unique tags.")
    print("Type 'help' for instructions, 'list' to browse, 'quit' to exit.\n")

    while True:
        try:
            raw = input("What do you feel like watching? > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if not raw:
            continue

        cmd = raw.lower()
        if cmd in {"quit", "exit", "q"}:
            print("Bye!")
            return
        if cmd == "help":
            print(HELP_TEXT)
            continue
        if cmd == "list":
            print("\nCatalog:")
            for m in movies:
                print(f"  - {m['title']} ({m['year']})  tags: {', '.join(m['tags'])}")
            print()
            continue
        if cmd.startswith("tags "):
            name = cmd[5:].strip()
            movie = index.get(name.lower())
            if movie:
                print(f"  {movie['title']} -> {', '.join(movie['tags'])}")
            else:
                print(f"  '{name}' not in catalog. Try 'list'.")
            continue

        # Two paths: title match OR raw tag list.
        movie_match = index.get(cmd)
        if movie_match:
            pref_vec = movie_to_vector(movie_match, vocab)
            header = (
                f"Because you liked '{movie_match['title']}', you might also enjoy:"
            )
            results = recommend(
                pref_vec, vocab, movies, vectors,
                exclude_title=movie_match["title"], top_n=5,
            )
        else:
            # Treat the whole input as a comma- or whitespace-separated tag list.
            tokens = re.split(r"[,\s]+", raw)
            pref_vec = tags_to_vector(tokens, vocab)
            nonzero = [vocab[i] for i, v in enumerate(pref_vec) if v]
            if not nonzero:
                print("  Couldn't parse any tags from that. Try 'help'.")
                continue
            header = f"Movies that match your taste for: {', '.join(nonzero)}"
            results = recommend(pref_vec, vocab, movies, vectors, top_n=5)

        print_recommendations(results, header)


# ---------------------------------------------------------------------------
# 7. Quick smoke test (run with: python movie_recommender.py --selftest)
# ---------------------------------------------------------------------------
def selftest() -> None:
    """Run a handful of canned queries and print the results so you can see
    the engine at work without typing anything."""
    movies = MOVIES
    vocab = build_vocab(movies)
    vectors = [movie_to_vector(m, vocab) for m in movies]
    index = build_movie_index(movies)

    cases = [
        ("Inception", "title"),
        ("sci-fi, mind-bending, emotional", "tags"),
        ("The Dark Knight", "title"),
        ("animation, family, emotional", "tags"),
    ]

    for raw, kind in cases:
        if kind == "title":
            seed = index[raw.lower()]
            pref = movie_to_vector(seed, vocab)
            results = recommend(
                pref, vocab, movies, vectors, exclude_title=seed["title"], top_n=3,
            )
            print_recommendations(
                results, f"Top 3 picks similar to '{seed['title']}':"
            )
        else:
            tokens = re.split(r"[,\s]+", raw)
            pref = tags_to_vector(tokens, vocab)
            results = recommend(pref, vocab, movies, vectors, top_n=3)
            print_recommendations(results, f"Top 3 picks for tags [{raw}]:")


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        selftest()
    else:
        interactive_loop()
