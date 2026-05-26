"""Streamlit dashboard for semantic review search."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA


BASE_DIR = Path(__file__).resolve().parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
DEFAULT_MODEL = "BAAI/bge-m3"


st.set_page_config(
    page_title="Review Embedding Dashboard",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(128, 128, 128, 0.08);
    }
    [data-testid="stMetric"] label,
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: inherit;
    }
    .hint {
        border: 1px solid rgba(59, 130, 246, 0.35);
        border-radius: 8px;
        background: rgba(59, 130, 246, 0.12);
        padding: 12px 14px;
        margin: 0.5rem 0 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def format_for_query(model_name: str, query: str) -> str:
    if "multilingual-e5" in model_name.lower():
        return f"query: {query}"
    return query


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    manifest_path = EMBEDDINGS_DIR / "manifest.json"
    if not manifest_path.exists():
        return {
            "model_name": DEFAULT_MODEL,
            "metadata_file": "review_metadata.parquet",
            "embeddings_file": "review_embeddings.npy",
        }
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner="Loading review vectors...")
def load_embedding_data(metadata_file: str, embeddings_file: str) -> tuple[pd.DataFrame, np.ndarray]:
    metadata_path = EMBEDDINGS_DIR / metadata_file
    embeddings_path = EMBEDDINGS_DIR / embeddings_file

    if not metadata_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError

    metadata = pd.read_parquet(metadata_path)
    embeddings = np.load(embeddings_path)

    if len(metadata) != len(embeddings):
        raise ValueError(
            f"Metadata rows ({len(metadata)}) do not match embeddings ({len(embeddings)})."
        )

    return metadata, embeddings


@st.cache_resource(show_spinner="Loading embedding model...")
def load_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def build_projection(embeddings: np.ndarray, review_ids: tuple[int, ...]) -> pd.DataFrame:
    del review_ids
    if len(embeddings) < 3:
        return pd.DataFrame(columns=["x", "y"])

    sample_size = min(len(embeddings), 2500)
    sample = embeddings[:sample_size]
    projection = PCA(n_components=2, random_state=42).fit_transform(sample)
    return pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1]})


def render_missing_embeddings() -> None:
    st.title("Review Embedding Dashboard")
    st.markdown(
        """
        <div class="hint">
        Embeddings have not been generated yet. Run the command below first,
        then refresh this dashboard.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python create_review_embeddings.py", language="bash")
    st.stop()


def filter_reviews(df: pd.DataFrame) -> pd.Series:
    with st.sidebar:
        st.header("Filters")

        categories = sorted(df["category"].dropna().unique().tolist())
        selected_categories = st.multiselect("Category", categories, default=categories)

        provinces = sorted(df["province"].dropna().unique().tolist())
        selected_provinces = st.multiselect("Province", provinces)

        sentiments = sorted(df["sentiment"].dropna().unique().tolist())
        selected_sentiments = st.multiselect("Sentiment", sentiments)

        min_rating, max_rating = st.slider(
            "Review rating",
            min_value=1,
            max_value=5,
            value=(1, 5),
        )

        top_k = st.slider("Search results", min_value=5, max_value=50, value=15, step=5)

    st.session_state["top_k"] = top_k

    mask = pd.Series(True, index=df.index)
    if selected_categories:
        mask &= df["category"].isin(selected_categories)
    if selected_provinces:
        mask &= df["province"].isin(selected_provinces)
    if selected_sentiments:
        mask &= df["sentiment"].isin(selected_sentiments)
    mask &= df["review_rating"].between(min_rating, max_rating)

    return mask


def render_overview(filtered_df: pd.DataFrame) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviews", f"{len(filtered_df):,}")
    metric_cols[1].metric("Places", f"{filtered_df['place_name'].nunique():,}")
    metric_cols[2].metric("Provinces", f"{filtered_df['province'].nunique():,}")
    metric_cols[3].metric("Avg rating", f"{filtered_df['review_rating'].mean():.2f}")

    chart_cols = st.columns((1.1, 1))

    with chart_cols[0]:
        rating_counts = (
            filtered_df["review_rating"].value_counts().rename_axis("rating").reset_index(name="reviews")
        )
        rating_counts = rating_counts.sort_values("rating")
        fig = px.bar(
            rating_counts,
            x="rating",
            y="reviews",
            color="reviews",
            color_continuous_scale=["#94a3b8", "#2563eb"],
            labels={"rating": "Rating", "reviews": "Reviews"},
            title="Review Rating Distribution",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")

    with chart_cols[1]:
        sentiment_counts = (
            filtered_df["sentiment"].fillna("unknown").value_counts().rename_axis("sentiment").reset_index(name="reviews")
        )
        fig = px.pie(
            sentiment_counts,
            names="sentiment",
            values="reviews",
            hole=0.55,
            title="Sentiment Mix",
            color_discrete_sequence=["#16a34a", "#64748b", "#dc2626", "#2563eb"],
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")


def render_search(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    mask: pd.Series,
    model_name: str,
) -> None:
    st.subheader("Semantic Search")
    query = st.text_input(
        "Search by meaning",
        placeholder="เช่น ร้านอาหารติดทะเล บรรยากาศดี ราคาไม่แพง",
    )

    filtered_df = df[mask].copy()
    filtered_embeddings = embeddings[mask.to_numpy()]

    if filtered_df.empty:
        st.warning("No reviews match the current filters.")
        return

    if not query.strip():
        st.dataframe(
            filtered_df[
                [
                    "category",
                    "place_name",
                    "province",
                    "review_rating",
                    "sentiment",
                    "review_text",
            ]
        ].head(100),
            width="stretch",
            hide_index=True,
        )
        return

    model = load_model(model_name)
    query_text = format_for_query(model_name, query.strip())
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
    scores = filtered_embeddings @ query_embedding

    result_df = filtered_df.copy()
    result_df["similarity"] = scores
    result_df = result_df.sort_values("similarity", ascending=False).head(st.session_state["top_k"])

    st.dataframe(
        result_df[
            [
                "similarity",
                "category",
                "place_name",
                "province",
                "review_rating",
                "sentiment",
                "review_text",
            ]
        ],
        width="stretch",
        hide_index=True,
        column_config={
            "similarity": st.column_config.ProgressColumn(
                "Similarity",
                min_value=0,
                max_value=1,
                format="%.3f",
            ),
            "review_text": st.column_config.TextColumn("Review", width="large"),
        },
    )

    st.subheader("Top Matching Places")
    place_scores = (
        result_df.groupby(["place_name", "province", "category"], as_index=False)
        .agg(
            avg_similarity=("similarity", "mean"),
            matched_reviews=("review_text", "count"),
            avg_rating=("review_rating", "mean"),
        )
        .sort_values(["avg_similarity", "matched_reviews"], ascending=[False, False])
    )
    st.dataframe(place_scores, width="stretch", hide_index=True)


def render_projection(df: pd.DataFrame, embeddings: np.ndarray) -> None:
    if len(df) < 3:
        return

    st.subheader("Embedding Map")
    projection = build_projection(embeddings, tuple(df["review_id"].head(2500).astype(int)))
    if projection.empty:
        return

    plot_df = df.head(len(projection)).copy()
    plot_df[["x", "y"]] = projection[["x", "y"]]
    plot_df["short_review"] = plot_df["review_text"].astype(str).str.slice(0, 140)

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="sentiment",
        symbol="category",
        hover_data={
            "place_name": True,
            "province": True,
            "review_rating": True,
            "short_review": True,
            "x": False,
            "y": False,
        },
        title="PCA Projection of Review Embeddings",
        color_discrete_map={
            "positive": "#16a34a",
            "neutral": "#64748b",
            "negative": "#dc2626",
        },
    )
    fig.update_traces(marker=dict(size=7, opacity=0.68))
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    manifest = load_manifest()

    try:
        df, embeddings = load_embedding_data(
            manifest["metadata_file"],
            manifest["embeddings_file"],
        )
    except FileNotFoundError:
        render_missing_embeddings()

    model_name = manifest.get("model_name", DEFAULT_MODEL)

    st.title("Review Embedding Dashboard")
    st.caption(f"Model: {model_name} | Embeddings: {len(df):,} reviews")

    mask = filter_reviews(df)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.warning("No reviews match the current filters.")
        return

    render_overview(filtered_df)

    tab_search, tab_map, tab_data = st.tabs(["Search", "Embedding Map", "Data"])
    with tab_search:
        render_search(df, embeddings, mask, model_name)
    with tab_map:
        render_projection(filtered_df.reset_index(drop=True), embeddings[mask.to_numpy()])
    with tab_data:
        st.dataframe(filtered_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
