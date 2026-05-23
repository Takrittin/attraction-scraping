#!/usr/bin/env python3
"""
Scrape tourist attractions in Southern Thailand with Google Places API and
classify review sentiment with a multilingual HuggingFace model.

Setup:
    pip install -r requirements.txt
    cp .env.local.example .env.local
    # Then put your real Google Places API key in .env.local

Example:
    python scrape_southern_thailand_attractions.py --output-dir output

Notes:
    - Google Places Details API returns only the reviews made available by
      Google for a place, often up to 5 reviews per place.
    - CSV files are written with utf-8-sig encoding for Thai compatibility.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: pandas. Install project dependencies with: "
        "python3 -m pip install -r requirements.txt"
    ) from exc

try:
    import requests
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: requests. Install project dependencies with: "
        "python3 -m pip install -r requirements.txt"
    ) from exc


API_DELAY_SECONDS = 1.0
NEXT_PAGE_DELAY_SECONDS = 2.0
GOOGLE_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
GOOGLE_PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places"
TEXT_SEARCH_FIELD_MASK = "places.id,nextPageToken"
PLACE_DETAILS_FIELD_MASK = (
    "id,displayName,formattedAddress,rating,userRatingCount,location,types,reviews"
)
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
PLACE_COLUMNS = [
    "name",
    "province",
    "province_thai",
    "address",
    "rating",
    "user_ratings_total",
    "lat",
    "lng",
    "place_id",
    "types",
]
REVIEW_COLUMNS = [
    "place_name",
    "province",
    "author_name",
    "review_rating",
    "review_text",
    "time",
]


def load_env_file(path: Path = Path(".env.local")) -> None:
    """Load simple KEY=VALUE pairs from a local env file without extra dependencies."""
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


@dataclass(frozen=True)
class Province:
    name: str
    thai_name: str


SOUTHERN_PROVINCES = [
    Province("Chumphon", "ชุมพร"),
    Province("Surat Thani", "สุราษฎร์ธานี"),
    Province("Ranong", "ระนอง"),
    Province("Nakhon Si Thammarat", "นครศรีธรรมราช"),
    Province("Krabi", "กระบี่"),
    Province("Phang Nga", "พังงา"),
    Province("Phuket", "ภูเก็ต"),
    Province("Trang", "ตรัง"),
    Province("Phatthalung", "พัทลุง"),
    Province("Satun", "สตูล"),
    Province("Songkhla", "สงขลา"),
    Province("Pattani", "ปัตตานี"),
    Province("Yala", "ยะลา"),
    Province("Narathiwat", "นราธิวาส"),
]


def parse_args() -> argparse.Namespace:
    load_env_file()
    parser = argparse.ArgumentParser(
        description="Scrape and analyze tourist attraction reviews in Southern Thailand."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("GOOGLE_PLACES_API_KEY"),
        help="Google Places API key. Defaults to GOOGLE_PLACES_API_KEY environment variable.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where CSV files will be saved.",
    )
    parser.add_argument(
        "--max-pages-per-province",
        type=int,
        default=3,
        help="Maximum Google Text Search result pages per province. Google usually caps this at 3.",
    )
    parser.add_argument(
        "--max-places-per-province",
        type=int,
        default=None,
        help="Optional cap for places per province, useful while testing.",
    )
    parser.add_argument(
        "--language",
        default="th",
        help="Preferred language for Google Places results and reviews.",
    )
    return parser.parse_args()


def call_google_api(
    method: str,
    url: str,
    api_key: str,
    field_mask: str,
    *,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Call a Places API (New) endpoint with error handling and rate-limit delay."""
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": field_mask,
    }

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        error_detail = ""
        if getattr(exc, "response", None) is not None:
            try:
                error_payload = exc.response.json()
                error_detail = f" - {error_payload.get('error', {}).get('message', '')}"
            except ValueError:
                error_detail = f" - {exc.response.text[:300]}"
        print(f"⚠️ Google API request failed: {exc}{error_detail}")
    except ValueError as exc:
        print(f"⚠️ Failed to parse API response JSON: {exc}")
    finally:
        time.sleep(API_DELAY_SECONDS)

    return None


def fetch_places_for_province(
    province: Province,
    api_key: str,
    language: str,
    max_pages: int,
    max_places: int | None,
) -> list[dict[str, Any]]:
    """Fetch tourist attraction candidates for one province using Text Search (New)."""
    places: list[dict[str, Any]] = []
    seen_place_ids: set[str] = set()
    next_page_token: str | None = None
    page = 0

    while page < max_pages:
        body = {
            "textQuery": f"tourist attractions in {province.name} Thailand",
            "includedType": "tourist_attraction",
            "languageCode": language,
            "pageSize": 20,
        }

        if next_page_token:
            # Google may need a short warm-up before pageToken becomes valid.
            time.sleep(NEXT_PAGE_DELAY_SECONDS)
            body["pageToken"] = next_page_token

        payload = call_google_api(
            method="POST",
            url=GOOGLE_TEXT_SEARCH_URL,
            api_key=api_key,
            field_mask=TEXT_SEARCH_FIELD_MASK,
            json_body=body,
        )
        if not payload:
            break

        page += 1
        for place in payload.get("places", []):
            place_id = place.get("id")
            if not place_id or place_id in seen_place_ids:
                continue

            seen_place_ids.add(place_id)
            places.append(place)

            if max_places is not None and len(places) >= max_places:
                return places

        next_page_token = payload.get("nextPageToken")
        if not next_page_token:
            break

    return places


def fetch_place_details(
    place_id: str,
    api_key: str,
    language: str,
) -> dict[str, Any] | None:
    """Fetch detailed place data, including address, rating, geometry, types, and reviews."""
    url = f"{GOOGLE_PLACE_DETAILS_URL}/{place_id}?languageCode={language}"
    return call_google_api(
        method="GET",
        url=url,
        api_key=api_key,
        field_mask=PLACE_DETAILS_FIELD_MASK,
    )


def extract_place_row(details: dict[str, Any], province: Province) -> dict[str, Any]:
    """Convert a Places Details result into the requested attraction row shape."""
    location = details.get("location", {})
    return {
        "name": details.get("displayName", {}).get("text", ""),
        "province": province.name,
        "province_thai": province.thai_name,
        "address": details.get("formattedAddress", ""),
        "rating": details.get("rating"),
        "user_ratings_total": details.get("userRatingCount"),
        "lat": location.get("latitude"),
        "lng": location.get("longitude"),
        "place_id": details.get("id", ""),
        "types": ", ".join(details.get("types", [])),
    }


def extract_review_rows(
    details: dict[str, Any],
    province: Province,
) -> list[dict[str, Any]]:
    """Convert available Google review objects into the requested review row shape."""
    place_name = details.get("displayName", {}).get("text", "")
    rows = []

    for review in details.get("reviews", []) or []:
        review_text = review.get("text", {}).get("text", "")
        author_name = review.get("authorAttribution", {}).get("displayName", "")
        rows.append(
            {
                "place_name": place_name,
                "province": province.name,
                "author_name": author_name,
                "review_rating": review.get("rating"),
                "review_text": review_text,
                "time": review.get("publishTime"),
            }
        )

    return rows


def load_sentiment_model() -> tuple[Any, Any, Any, Any]:
    """Load the HuggingFace tokenizer and model once."""
    print(f"🤗 Loading sentiment model: {SENTIMENT_MODEL_NAME}")
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing sentiment dependencies. Install them with: "
            "python3 -m pip install -r requirements.txt"
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME, use_fast=False)
    except Exception as exc:
        raise SystemExit(
            "Failed to load the HuggingFace tokenizer. Install tokenizer dependencies "
            "with: python3 -m pip install sentencepiece tiktoken"
        ) from exc

    model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model, device, torch


def fallback_sentiment_from_rating(review_rating: Any) -> tuple[str, float]:
    """Classify empty reviews using the fallback star-rating rule."""
    try:
        rating = float(review_rating)
    except (TypeError, ValueError):
        return "neutral", 0.0

    if rating >= 4:
        return "positive", 1.0
    if rating == 3:
        return "neutral", 1.0
    return "negative", 1.0


def classify_text_sentiment(
    text: str,
    tokenizer: Any,
    model: Any,
    device: Any,
    torch_module: Any,
) -> tuple[str, float]:
    """Classify a non-empty review into positive, neutral, or negative."""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch_module.no_grad():
        logits = model(**encoded).logits
        probabilities = torch_module.softmax(logits, dim=1)[0]

    confidence, label_index = torch_module.max(probabilities, dim=0)
    raw_label = model.config.id2label.get(int(label_index), f"LABEL_{int(label_index)}")
    normalized_label = normalize_sentiment_label(raw_label)
    return normalized_label, float(confidence.item())


def normalize_sentiment_label(raw_label: str) -> str:
    """Normalize model labels such as LABEL_0 into sentiment names."""
    label = raw_label.lower()
    label_map = {
        "label_0": "negative",
        "label_1": "neutral",
        "label_2": "positive",
        "negative": "negative",
        "neutral": "neutral",
        "positive": "positive",
    }
    return label_map.get(label, "neutral")


def add_sentiment_columns(reviews_df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment and confidence columns to all review rows."""
    if reviews_df.empty:
        reviews_df["sentiment"] = pd.Series(dtype="object")
        reviews_df["confidence"] = pd.Series(dtype="float")
        return reviews_df

    tokenizer, model, device, torch_module = load_sentiment_model()
    sentiments = []
    confidences = []

    for row_number, row in enumerate(reviews_df.itertuples(index=False), start=1):
        text = str(getattr(row, "review_text") or "").strip()
        rating = getattr(row, "review_rating")

        if not text:
            sentiment, confidence = fallback_sentiment_from_rating(rating)
        else:
            try:
                sentiment, confidence = classify_text_sentiment(
                    text, tokenizer, model, device, torch_module
                )
            except Exception as exc:  # Model inference should not stop the whole scrape.
                print(f"⚠️ Sentiment failed for review {row_number}: {exc}")
                sentiment, confidence = fallback_sentiment_from_rating(rating)

        sentiments.append(sentiment)
        confidences.append(confidence)

        if row_number % 50 == 0:
            print(f"  Analyzed {row_number}/{len(reviews_df)} reviews...")

    reviews_df = reviews_df.copy()
    reviews_df["sentiment"] = sentiments
    reviews_df["confidence"] = confidences
    return reviews_df


def scrape_all_data(
    api_key: str,
    language: str,
    max_pages_per_province: int,
    max_places_per_province: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scrape places and reviews across all configured Southern Thailand provinces."""
    place_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []

    for index, province in enumerate(SOUTHERN_PROVINCES, start=1):
        print(
            f"📍 Fetching {province.name} ({province.thai_name})... "
            f"({index}/{len(SOUTHERN_PROVINCES)} provinces)"
        )
        search_results = fetch_places_for_province(
            province=province,
            api_key=api_key,
            language=language,
            max_pages=max_pages_per_province,
            max_places=max_places_per_province,
        )
        print(f"  Found {len(search_results)} place candidates.")

        for place_index, place in enumerate(search_results, start=1):
            place_id = place.get("id")
            if not place_id:
                continue

            details = fetch_place_details(place_id, api_key, language)
            if not details:
                continue

            place_rows.append(extract_place_row(details, province))
            review_rows.extend(extract_review_rows(details, province))
            print(
                f"  [{place_index}/{len(search_results)}] "
                f"{details.get('displayName', {}).get('text', 'Unknown place')} - "
                f"{len(details.get('reviews', []) or [])} reviews"
            )

    places_df = pd.DataFrame(place_rows, columns=PLACE_COLUMNS)
    reviews_df = pd.DataFrame(review_rows, columns=REVIEW_COLUMNS)
    return places_df, reviews_df


def build_attraction_summary(
    places_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one summary row per attraction with review sentiment counts."""
    summary_columns = [
        "place_name",
        "province",
        "province_thai",
        "lat",
        "lng",
        "avg_rating",
        "total_reviews",
        "positive_count",
        "negative_count",
        "neutral_count",
    ]

    if places_df.empty:
        return pd.DataFrame(columns=summary_columns)

    base_summary = places_df.rename(
        columns={
            "name": "place_name",
            "rating": "avg_rating",
            "user_ratings_total": "total_reviews",
        }
    )[
        [
            "place_name",
            "province",
            "province_thai",
            "lat",
            "lng",
            "avg_rating",
            "total_reviews",
        ]
    ]

    if reviews_df.empty:
        for column in ["positive_count", "negative_count", "neutral_count"]:
            base_summary[column] = 0
        return base_summary[summary_columns]

    sentiment_counts = (
        reviews_df.groupby(["place_name", "province", "sentiment"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename_axis(None, axis=1)
    )

    for sentiment in ["positive", "negative", "neutral"]:
        if sentiment not in sentiment_counts.columns:
            sentiment_counts[sentiment] = 0

    sentiment_counts = sentiment_counts.rename(
        columns={
            "positive": "positive_count",
            "negative": "negative_count",
            "neutral": "neutral_count",
        }
    )

    summary_df = base_summary.merge(
        sentiment_counts[
            [
                "place_name",
                "province",
                "positive_count",
                "negative_count",
                "neutral_count",
            ]
        ],
        on=["place_name", "province"],
        how="left",
    )

    count_columns = ["positive_count", "negative_count", "neutral_count"]
    summary_df[count_columns] = summary_df[count_columns].fillna(0).astype(int)
    return summary_df[summary_columns]


def save_outputs(
    reviews_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save all requested CSV outputs with utf-8-sig encoding."""
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "all_reviews.csv": reviews_df,
        "positive_reviews.csv": reviews_df[reviews_df["sentiment"] == "positive"],
        "negative_reviews.csv": reviews_df[reviews_df["sentiment"] == "negative"],
        "neutral_reviews.csv": reviews_df[reviews_df["sentiment"] == "neutral"],
        "attraction_summary.csv": summary_df,
    }

    for filename, dataframe in outputs.items():
        path = output_dir / filename
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"💾 Saved {path} ({len(dataframe)} rows)")


def save_raw_checkpoint(
    places_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Save raw scraped data before sentiment analysis starts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_places_path = output_dir / "raw_attractions.csv"
    raw_reviews_path = output_dir / "raw_reviews.csv"
    places_df.to_csv(raw_places_path, index=False, encoding="utf-8-sig")
    reviews_df.to_csv(raw_reviews_path, index=False, encoding="utf-8-sig")
    print(f"💾 Saved {raw_places_path} ({len(places_df)} rows)")
    print(f"💾 Saved {raw_reviews_path} ({len(reviews_df)} rows)")


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit(
            "Missing Google Places API key. Set GOOGLE_PLACES_API_KEY or pass --api-key."
        )
    output_dir = Path(args.output_dir)

    places_df, reviews_df = scrape_all_data(
        api_key=args.api_key,
        language=args.language,
        max_pages_per_province=args.max_pages_per_province,
        max_places_per_province=args.max_places_per_province,
    )

    print(f"✅ Collected {len(places_df)} attractions and {len(reviews_df)} reviews.")
    save_raw_checkpoint(places_df, reviews_df, output_dir)
    reviews_df = add_sentiment_columns(reviews_df)
    summary_df = build_attraction_summary(places_df, reviews_df)
    save_outputs(reviews_df, summary_df, output_dir)
    print("✅ Done.")


if __name__ == "__main__":
    main()
