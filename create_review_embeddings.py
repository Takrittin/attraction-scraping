"""Create sentence embeddings for cleaned attraction and restaurant reviews."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "BAAI/bge-m3"
REVIEW_FILES = {
    "restaurant": BASE_DIR / "restaurant_output" / "cleaned" / "all_reviews.csv",
    "attraction": BASE_DIR / "attraction_output" / "cleaned" / "all_reviews.csv",
}


def read_reviews() -> pd.DataFrame:
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

    return reviews


def e5_passages(texts: pd.Series) -> list[str]:
    """E5 models expect a passage/query prefix for retrieval quality."""
    return [f"passage: {text}" for text in texts.tolist()]


def encode_texts(
    model: SentenceTransformer,
    model_name: str,
    texts: pd.Series,
    batch_size: int,
) -> np.ndarray:
    if "multilingual-e5" in model_name.lower():
        values = e5_passages(texts)
    else:
        values = texts.tolist()

    embeddings = model.encode(
        values,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype="float32")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate sentence embeddings for cleaned review CSV files."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"SentenceTransformer model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASE_DIR / "embeddings",
        help="Directory for metadata, vectors, and manifest files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews = read_reviews()
    print(f"Loaded {len(reviews):,} cleaned reviews.")
    print(f"Loading embedding model: {args.model}")

    model = SentenceTransformer(args.model)
    embeddings = encode_texts(model, args.model, reviews["review_text"], args.batch_size)

    metadata_path = output_dir / "review_metadata.parquet"
    embeddings_path = output_dir / "review_embeddings.npy"
    manifest_path = output_dir / "manifest.json"

    reviews.to_parquet(metadata_path, index=False)
    np.save(embeddings_path, embeddings)

    manifest = {
        "model_name": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "review_count": int(len(reviews)),
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
