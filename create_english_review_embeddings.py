"""Create one English summary vector per place.

This script groups cleaned Thai / mixed-language reviews by place, creates one
40-word English summary for each place, then encodes that summary with
sentence-transformers/all-MiniLM-L6-v2 by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
DEFAULT_VERTEX_LOCATION = "us-central1"
DEFAULT_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-th-en"
DEFAULT_MAX_REVIEWS_PER_PLACE = 20
REVIEW_FILES = {
    "restaurant": BASE_DIR / "restaurant_output" / "cleaned" / "all_reviews.csv",
    "attraction": BASE_DIR / "attraction_output" / "cleaned" / "all_reviews.csv",
}
SUMMARY_FILES = {
    "restaurant": BASE_DIR / "restaurant_output" / "cleaned" / "restaurant_summary.csv",
    "attraction": BASE_DIR / "attraction_output" / "cleaned" / "attraction_summary.csv",
}


def load_env_file(path: Path) -> None:
    """Load simple KEY=value pairs without adding a dotenv dependency."""
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def read_reviews(limit: int | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for category, csv_path in REVIEW_FILES.items():
        if not csv_path.exists():
            print(f"Skipping missing file: {csv_path}")
            continue

        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["category"] = category
        frame["source_file"] = str(csv_path.relative_to(BASE_DIR))
        frames.append(frame)

    if not frames:
        raise FileNotFoundError("No cleaned review CSV files were found.")

    reviews = pd.concat(frames, ignore_index=True)
    reviews = reviews.dropna(subset=["review_text"]).copy()
    reviews["review_text"] = reviews["review_text"].astype(str).str.strip()
    reviews = reviews[reviews["review_text"] != ""]
    reviews = reviews.drop_duplicates(
        subset=["category", "place_name", "author_name", "review_text"]
    )
    reviews = reviews.reset_index(drop=True)
    reviews.insert(0, "review_id", reviews.index.astype(int))
    reviews["original_review_text"] = reviews["review_text"]
    reviews["review_key"] = reviews.apply(make_review_key, axis=1)

    if limit:
        reviews = reviews.head(limit).copy()

    return reviews


def read_place_summaries() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for category, csv_path in SUMMARY_FILES.items():
        if not csv_path.exists():
            continue

        frame = pd.read_csv(csv_path, encoding="utf-8-sig")
        frame["category"] = category
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    summaries = pd.concat(frames, ignore_index=True)
    summaries = summaries.drop_duplicates(subset=["category", "place_name", "province"])
    return summaries


def make_place_key(row: pd.Series) -> str:
    parts = [
        str(row.get("category", "")),
        str(row.get("place_name", "")),
        str(row.get("province", "")),
    ]
    raw_key = "\u241f".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def build_place_rows(
    reviews: pd.DataFrame,
    max_reviews_per_place: int,
    limit: int | None = None,
) -> pd.DataFrame:
    place_rows: list[dict[str, Any]] = []
    group_columns = ["category", "place_name", "province"]

    for (category, place_name, province), group in reviews.groupby(group_columns, dropna=False):
        group = group.sort_values(["review_rating", "time"], ascending=[False, False])
        selected_reviews = group.head(max_reviews_per_place)
        review_lines = [
            f"- Rating {int(row.review_rating)}: {str(row.original_review_text)}"
            for row in selected_reviews.itertuples(index=False)
        ]
        sentiment_counts = group["sentiment"].value_counts(dropna=False).to_dict()

        place_rows.append(
            {
                "category": category,
                "place_name": place_name,
                "province": province,
                "review_count": int(len(group)),
                "sampled_review_count": int(len(selected_reviews)),
                "avg_review_rating": float(group["review_rating"].mean()),
                "positive_review_count": int(sentiment_counts.get("positive", 0)),
                "neutral_review_count": int(sentiment_counts.get("neutral", 0)),
                "negative_review_count": int(sentiment_counts.get("negative", 0)),
                "source_files": ", ".join(sorted(group["source_file"].dropna().unique())),
                "combined_review_text": "\n".join(review_lines),
            }
        )

    places = pd.DataFrame(place_rows)
    place_summaries = read_place_summaries()
    if not place_summaries.empty:
        summary_columns = [
            column
            for column in [
                "category",
                "place_name",
                "province",
                "province_thai",
                "lat",
                "lng",
                "avg_rating",
                "total_reviews",
            ]
            if column in place_summaries.columns
        ]
        places = places.merge(
            place_summaries[summary_columns],
            on=["category", "place_name", "province"],
            how="left",
        )
        places = places.rename(
            columns={
                "avg_rating": "google_avg_rating",
                "total_reviews": "google_total_reviews",
            }
        )

    places = places.sort_values(["category", "province", "place_name"]).reset_index(drop=True)
    places.insert(0, "place_id", places.index.astype(int))
    places["place_key"] = places.apply(make_place_key, axis=1)

    if limit:
        places = places.head(limit).copy()

    return places


def make_review_key(row: pd.Series) -> str:
    parts = [
        str(row.get("category", "")),
        str(row.get("place_name", "")),
        str(row.get("author_name", "")),
        str(row.get("review_rating", "")),
        str(row.get("time", "")),
        str(row.get("review_text", "")),
    ]
    raw_key = "\u241f".join(parts)
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).rstrip(".,;:") + "."


def clean_summary(text: str, max_words: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip('"').strip("'").strip()
    return limit_words(cleaned, max_words)


def merge_cached_summaries(
    places: pd.DataFrame,
    cache_path: Path,
    force_transform: bool,
) -> pd.DataFrame:
    places["english_40word_summary"] = ""

    if force_transform or not cache_path.exists():
        return places

    cached = pd.read_parquet(cache_path)
    if "place_key" not in cached.columns or "english_40word_summary" not in cached.columns:
        return places

    summary_map = (
        cached.dropna(subset=["english_40word_summary"])
        .drop_duplicates(subset=["place_key"], keep="last")
        .set_index("place_key")["english_40word_summary"]
    )
    places["english_40word_summary"] = (
        places["place_key"].map(summary_map).fillna("").astype(str)
    )
    return places


def save_metadata(metadata: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_parquet(path, index=False)


def parse_gemini_results(payload: dict[str, Any]) -> dict[int, str]:
    candidates = payload.get("candidates", [])
    if not candidates:
        raise KeyError(f"Gemini response has no candidates: {payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    content = " ".join(str(part.get("text", "")) for part in parts).strip()
    if not content:
        raise KeyError(f"Gemini response has no text content: {payload}")

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise
        parsed = json.loads(match.group(0))

    summaries: dict[int, str] = {}
    for item in parsed.get("results", []):
        try:
            place_id = int(item["id"])
            summary = str(
                item.get("english_40word_summary")
                or item.get("english_40word_review")
                or item["summary"]
            )
        except (KeyError, TypeError, ValueError):
            continue
        summaries[place_id] = summary
    return summaries


def get_gcloud_value(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["gcloud", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Could not read Google Cloud credentials with gcloud. Install the "
            "Google Cloud CLI and run `gcloud auth login` first."
        ) from exc
    return result.stdout.strip()


def get_vertex_project_id() -> str:
    project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        return project_id

    project_id = get_gcloud_value(["config", "get-value", "project"])
    if not project_id or project_id == "(unset)":
        raise RuntimeError(
            "Missing Google Cloud project. Set VERTEX_PROJECT_ID in .env.local "
            "or run `gcloud config set project YOUR_PROJECT_ID`."
        )
    return project_id


def get_vertex_access_token() -> str:
    token = os.getenv("VERTEX_ACCESS_TOKEN")
    if token:
        return token
    return get_gcloud_value(["auth", "print-access-token"])


def gemini_summarize_batch(
    rows: pd.DataFrame,
    model_name: str,
    max_words: int,
    timeout: int,
) -> dict[int, str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to .env.local or use another "
            "--summary-provider."
        )

    places = [
        {
            "id": int(row.place_id),
            "place_name": str(row.place_name),
            "category": str(row.category),
            "province": str(row.province),
            "review_count": int(row.review_count),
            "reviews": str(row.combined_review_text),
        }
        for row in rows.itertuples(index=False)
    ]

    prompt = {
        "instruction": (
            "Translate each place's Thai/mixed-language review evidence to "
            f"natural English and summarize the place in {max_words} words or "
            "fewer. Combine repeated opinions and keep important signals about "
            "food, service, price, atmosphere, location, cleanliness, and "
            "problems. Return valid JSON only in exactly "
            'this shape: {"results":[{"id":123,'
            '"english_40word_summary":"summary"}]}.'
        ),
        "places": places,
    }

    response = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": json.dumps(prompt, ensure_ascii=False)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1000].strip()
        raise requests.HTTPError(
            f"{exc}. Response body: {body}",
            response=response,
        ) from exc

    return parse_gemini_results(response.json())


def vertex_summarize_batch(
    rows: pd.DataFrame,
    model_name: str,
    max_words: int,
    timeout: int,
    project_id: str,
    location: str,
    access_token: str,
) -> dict[int, str]:
    places = [
        {
            "id": int(row.place_id),
            "place_name": str(row.place_name),
            "category": str(row.category),
            "province": str(row.province),
            "review_count": int(row.review_count),
            "reviews": str(row.combined_review_text),
        }
        for row in rows.itertuples(index=False)
    ]

    prompt = {
        "instruction": (
            "Translate each place's Thai/mixed-language review evidence to "
            f"natural English and summarize the place in {max_words} words or "
            "fewer. Combine repeated opinions and keep important signals about "
            "food, service, price, atmosphere, location, cleanliness, and "
            "problems. Return valid JSON only in exactly "
            'this shape: {"results":[{"id":123,'
            '"english_40word_summary":"summary"}]}.'
        ),
        "places": places,
    }

    response = requests.post(
        (
            f"https://{location}-aiplatform.googleapis.com/v1/projects/"
            f"{project_id}/locations/{location}/publishers/google/models/"
            f"{model_name}:generateContent"
        ),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": json.dumps(prompt, ensure_ascii=False)},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.1,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout,
    )

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        body = response.text[:1000].strip()
        raise requests.HTTPError(
            f"{exc}. Response body: {body}",
            response=response,
        ) from exc

    return parse_gemini_results(response.json())


def transform_with_gemini(
    places: pd.DataFrame,
    metadata_path: Path,
    model_name: str,
    max_words: int,
    transform_batch_size: int,
    request_timeout: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    pending_mask = places["english_40word_summary"].str.strip() == ""
    pending = places[pending_mask]
    print(f"Places needing English summaries: {len(pending):,}")

    for start in range(0, len(pending), transform_batch_size):
        batch = pending.iloc[start : start + transform_batch_size]
        summaries: dict[int, str] = {}

        for attempt in range(1, 4):
            try:
                print(f"Summarizing with Gemini model: {model_name}")
                summaries = gemini_summarize_batch(
                    batch,
                    model_name=model_name,
                    max_words=max_words,
                    timeout=request_timeout,
                )
                break
            except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"Gemini summary batch failed after 3 attempts: {exc}"
                    ) from exc
                wait = 2**attempt
                print(f"Batch failed, retrying in {wait}s: {exc}")
                time.sleep(wait)

        for place_id, summary in summaries.items():
            places.loc[
                places["place_id"] == place_id, "english_40word_summary"
            ] = clean_summary(summary, max_words)

        missing_ids = set(batch["place_id"].tolist()) - set(summaries)
        if missing_ids:
            print(f"Warning: {len(missing_ids)} summaries missing in this batch.")

        save_metadata(places, metadata_path)
        done = (places["english_40word_summary"].str.strip() != "").sum()
        print(f"Saved summary checkpoint: {done:,}/{len(places):,}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return places


def transform_with_vertex(
    places: pd.DataFrame,
    metadata_path: Path,
    model_name: str,
    max_words: int,
    transform_batch_size: int,
    request_timeout: int,
    sleep_seconds: float,
    project_id: str,
    location: str,
) -> pd.DataFrame:
    access_token = get_vertex_access_token()
    pending_mask = places["english_40word_summary"].str.strip() == ""
    pending = places[pending_mask]
    print(f"Places needing English summaries: {len(pending):,}")
    print(f"Using Vertex AI project: {project_id}")
    print(f"Using Vertex AI location: {location}")

    for start in range(0, len(pending), transform_batch_size):
        batch = pending.iloc[start : start + transform_batch_size]
        summaries: dict[int, str] = {}

        for attempt in range(1, 4):
            try:
                print(f"Summarizing with Vertex AI model: {model_name}")
                summaries = vertex_summarize_batch(
                    batch,
                    model_name=model_name,
                    max_words=max_words,
                    timeout=request_timeout,
                    project_id=project_id,
                    location=location,
                    access_token=access_token,
                )
                break
            except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
                if attempt == 3:
                    raise RuntimeError(
                        f"Vertex AI summary batch failed after 3 attempts: {exc}"
                    ) from exc
                wait = 2**attempt
                print(f"Batch failed, retrying in {wait}s: {exc}")
                time.sleep(wait)

        for place_id, summary in summaries.items():
            places.loc[
                places["place_id"] == place_id, "english_40word_summary"
            ] = clean_summary(summary, max_words)

        missing_ids = set(batch["place_id"].tolist()) - set(summaries)
        if missing_ids:
            print(f"Warning: {len(missing_ids)} summaries missing in this batch.")

        save_metadata(places, metadata_path)
        done = (places["english_40word_summary"].str.strip() != "").sum()
        print(f"Saved summary checkpoint: {done:,}/{len(places):,}")

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return places


def contains_thai(text: str) -> bool:
    return bool(re.search(r"[\u0E00-\u0E7F]", text))


def transform_with_transformers(
    places: pd.DataFrame,
    metadata_path: Path,
    translation_model: str,
    max_words: int,
    transform_batch_size: int,
) -> pd.DataFrame:
    from transformers import pipeline

    print(f"Loading translation model: {translation_model}")
    translator = pipeline("translation", model=translation_model)
    pending_mask = places["english_40word_summary"].str.strip() == ""
    pending = places[pending_mask]
    print(f"Places needing English summaries: {len(pending):,}")

    for start in range(0, len(pending), transform_batch_size):
        batch = pending.iloc[start : start + transform_batch_size]
        texts = batch["combined_review_text"].astype(str).tolist()
        translated: list[str] = []

        thai_positions: list[int] = []
        thai_texts: list[str] = []
        for idx, text in enumerate(texts):
            if contains_thai(text):
                thai_positions.append(idx)
                thai_texts.append(text)
                translated.append("")
            else:
                translated.append(text)

        if thai_texts:
            outputs = translator(thai_texts, max_length=160)
            for idx, output in zip(thai_positions, outputs):
                translated[idx] = output["translation_text"]

        for row, text in zip(batch.itertuples(index=False), translated):
            places.loc[
                places["place_id"] == row.place_id, "english_40word_summary"
            ] = clean_summary(text, max_words)

        save_metadata(places, metadata_path)
        done = (places["english_40word_summary"].str.strip() != "").sum()
        print(f"Saved summary checkpoint: {done:,}/{len(places):,}")

    return places


def encode_summaries(
    places: pd.DataFrame,
    embedding_model: str,
    batch_size: int,
) -> np.ndarray:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing package: sentence-transformers. Install project "
            "dependencies with `python3 -m pip install -r requirements.txt`, "
            "then run this script again."
        ) from exc

    texts = places["english_40word_summary"].fillna("").astype(str).tolist()
    print(f"Loading embedding model: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate/summarize cleaned reviews into <=40-word English text "
            "and embed them with all-MiniLM-L6-v2. The default provider is "
            "Vertex AI with gemini-2.5-flash-lite."
        )
    )
    parser.add_argument(
        "--summary-provider",
        choices=["vertex", "gemini", "transformers", "existing"],
        default="vertex",
        help=(
            "How to create English summaries. 'existing' only embeds summaries "
            "already saved in the metadata file."
        ),
    )
    parser.add_argument(
        "--vertex-project-id",
        default=os.getenv("VERTEX_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"),
        help=(
            "Google Cloud project ID for Vertex AI. If omitted, the script "
            "uses `gcloud config get-value project`."
        ),
    )
    parser.add_argument(
        "--vertex-location",
        default=os.getenv("VERTEX_LOCATION", DEFAULT_VERTEX_LOCATION),
        help=f"Vertex AI location. Default: {DEFAULT_VERTEX_LOCATION}",
    )
    parser.add_argument(
        "--gemini-model",
        default=os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL),
        help=(
            "Gemini model for Vertex AI or Gemini API translation/summarization. "
            f"Default: {DEFAULT_GEMINI_MODEL}"
        ),
    )
    parser.add_argument(
        "--translation-model",
        default=DEFAULT_TRANSLATION_MODEL,
        help=(
            "Hugging Face translation model used with --summary-provider "
            f"transformers. Default: {DEFAULT_TRANSLATION_MODEL}"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model. Default: {DEFAULT_EMBEDDING_MODEL}",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=40,
        help="Maximum words in each English summary.",
    )
    parser.add_argument(
        "--max-reviews-per-place",
        type=int,
        default=DEFAULT_MAX_REVIEWS_PER_PLACE,
        help=(
            "Maximum review texts to send to the summarization model for each "
            f"place. Default: {DEFAULT_MAX_REVIEWS_PER_PLACE}."
        ),
    )
    parser.add_argument(
        "--transform-batch-size",
        type=int,
        default=12,
        help="Number of places per translation/summarization batch.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=64,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Only create/save place summaries; do not generate vector embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "embeddings",
        help="Directory for metadata, vectors, and manifest files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N places. Useful for testing.",
    )
    parser.add_argument(
        "--force-transform",
        action="store_true",
        help="Ignore cached summaries and transform every place again.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=90,
        help="Chat API request timeout in seconds.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.0,
        help="Optional sleep between chat API batches for rate limits.",
    )
    return parser.parse_args()


def main() -> None:
    load_env_file(BASE_DIR / ".env.local")
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "place_metadata_english_40words.parquet"
    embeddings_path = output_dir / "place_embeddings_english_40words.npy"
    manifest_path = output_dir / "manifest_place_english_40words.json"

    reviews = read_reviews()
    places = build_place_rows(
        reviews,
        max_reviews_per_place=args.max_reviews_per_place,
        limit=args.limit,
    )
    places = merge_cached_summaries(places, metadata_path, args.force_transform)
    print(f"Loaded {len(reviews):,} cleaned reviews.")
    print(f"Built {len(places):,} place rows.")

    if args.summary_provider == "vertex":
        project_id = args.vertex_project_id or get_vertex_project_id()
        places = transform_with_vertex(
            places=places,
            metadata_path=metadata_path,
            model_name=args.gemini_model,
            max_words=args.max_words,
            transform_batch_size=args.transform_batch_size,
            request_timeout=args.request_timeout,
            sleep_seconds=args.sleep_seconds,
            project_id=project_id,
            location=args.vertex_location,
        )
    elif args.summary_provider == "gemini":
        places = transform_with_gemini(
            places=places,
            metadata_path=metadata_path,
            model_name=args.gemini_model,
            max_words=args.max_words,
            transform_batch_size=args.transform_batch_size,
            request_timeout=args.request_timeout,
            sleep_seconds=args.sleep_seconds,
        )
    elif args.summary_provider == "transformers":
        places = transform_with_transformers(
            places=places,
            metadata_path=metadata_path,
            translation_model=args.translation_model,
            max_words=args.max_words,
            transform_batch_size=args.transform_batch_size,
        )
    else:
        missing = (places["english_40word_summary"].str.strip() == "").sum()
        if missing:
            raise RuntimeError(
                f"{missing:,} places do not have english_40word_summary yet. "
                "Run with --summary-provider vertex, gemini, or transformers first."
            )

    places["summary_word_count"] = places["english_40word_summary"].apply(word_count)
    too_long = places["summary_word_count"] > args.max_words
    if too_long.any():
        places.loc[too_long, "english_40word_summary"] = places.loc[
            too_long, "english_40word_summary"
        ].apply(lambda value: limit_words(value, args.max_words))
        places["summary_word_count"] = places["english_40word_summary"].apply(word_count)

    empty_summaries = (places["english_40word_summary"].str.strip() == "").sum()
    if empty_summaries:
        raise RuntimeError(
            f"{empty_summaries:,} places still have empty English summaries."
        )

    save_metadata(places, metadata_path)

    if args.skip_embedding:
        print(f"Saved metadata: {metadata_path}")
        print("Skipped embedding generation because --skip-embedding was set.")
        return

    embeddings = encode_summaries(
        places,
        embedding_model=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )

    np.save(embeddings_path, embeddings)

    manifest = {
        "embedding_model": args.embedding_model,
        "summary_provider": args.summary_provider,
        "vertex_project_id": project_id if args.summary_provider == "vertex" else None,
        "vertex_location": (
            args.vertex_location if args.summary_provider == "vertex" else None
        ),
        "gemini_model": (
            args.gemini_model
            if args.summary_provider in {"vertex", "gemini"}
            else None
        ),
        "translation_model": (
            args.translation_model if args.summary_provider == "transformers" else None
        ),
        "summary_level": "place",
        "text_column": "english_40word_summary",
        "max_words": int(args.max_words),
        "max_reviews_per_place": int(args.max_reviews_per_place),
        "embedding_dim": int(embeddings.shape[1]),
        "review_count": int(len(reviews)),
        "place_count": int(len(places)),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata_file": metadata_path.name,
        "embeddings_file": embeddings_path.name,
        "normalized": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved embeddings: {embeddings_path}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
