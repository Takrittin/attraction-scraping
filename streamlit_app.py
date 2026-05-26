"""Streamlit dashboard for testing local embedding search."""

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
    page_title="Embedding Search Tester",
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


def manifest_model_name(manifest: dict) -> str:
    return manifest.get("model_name") or manifest.get("embedding_model") or DEFAULT_MODEL


def manifest_level(manifest: dict) -> str:
    return str(manifest.get("summary_level") or "review")


def manifest_label(manifest: dict, manifest_path: Path) -> str:
    level = manifest_level(manifest).title()
    model_name = manifest_model_name(manifest).split("/")[-1]
    rows = manifest.get("place_count") or manifest.get("review_count")
    row_text = f"{rows:,} rows" if isinstance(rows, int) else "available"
    return f"{level} embeddings - {model_name} ({row_text})"


@st.cache_data(show_spinner=False)
def load_manifests() -> list[dict]:
    manifests: list[dict] = []

    for manifest_path in sorted(EMBEDDINGS_DIR.glob("manifest*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        metadata_file = manifest.get("metadata_file")
        embeddings_file = manifest.get("embeddings_file")
        if not metadata_file or not embeddings_file:
            continue

        if not (EMBEDDINGS_DIR / metadata_file).exists():
            continue
        if not (EMBEDDINGS_DIR / embeddings_file).exists():
            continue

        manifest["_manifest_file"] = manifest_path.name
        manifest["_label"] = manifest_label(manifest, manifest_path)
        manifests.append(manifest)

    if manifests:
        return manifests

    return [
        {
            "_manifest_file": "manifest.json",
            "_label": "Review embeddings - BAAI/bge-m3",
            "model_name": DEFAULT_MODEL,
            "metadata_file": "review_metadata.parquet",
            "embeddings_file": "review_embeddings.npy",
            "summary_level": "review",
        }
    ]


@st.cache_data(show_spinner="Loading vectors...")
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
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        return SentenceTransformer(model_name)


@st.cache_data(show_spinner=False)
def build_projection(embeddings: np.ndarray, row_ids: tuple[int, ...]) -> pd.DataFrame:
    del row_ids
    if len(embeddings) < 3:
        return pd.DataFrame(columns=["x", "y"])

    sample_size = min(len(embeddings), 2500)
    sample = embeddings[:sample_size]
    projection = PCA(n_components=2, random_state=42).fit_transform(sample)
    return pd.DataFrame({"x": projection[:, 0], "y": projection[:, 1]})


def render_missing_embeddings() -> None:
    st.title("Embedding Search Tester")
    st.markdown(
        """
        <div class="hint">
        No matching metadata/vector files were found in the embeddings folder.
        Generate one of the embedding datasets below, then refresh this app.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("python create_review_embeddings.py", language="bash")
    st.code("python create_english_review_embeddings.py", language="bash")
    st.stop()


def resolve_text_column(df: pd.DataFrame, manifest: dict) -> str:
    preferred = manifest.get("text_column")
    candidates = [
        preferred,
        "review_text",
        "english_40word_summary",
        "combined_review_text",
    ]
    for column in candidates:
        if column and column in df.columns:
            return str(column)
    return str(df.columns[0])


def resolve_rating_column(df: pd.DataFrame) -> str | None:
    for column in ["review_rating", "avg_review_rating", "google_avg_rating", "avg_rating"]:
        if column in df.columns:
            return column
    return None


def resolve_id_column(df: pd.DataFrame) -> str | None:
    for column in ["review_id", "place_id"]:
        if column in df.columns:
            return column
    return None


def row_label(manifest: dict) -> str:
    return "Places" if manifest_level(manifest) == "place" else "Reviews"


def choose_manifest(manifests: list[dict]) -> dict:
    with st.sidebar:
        st.header("Dataset")
        options = [manifest["_label"] for manifest in manifests]
        selected = st.selectbox("Embedding file", options, index=0)
    return manifests[options.index(selected)]


def filter_rows(df: pd.DataFrame, manifest: dict, rating_column: str | None) -> pd.Series:
    with st.sidebar:
        st.header("Filters")

        selected_categories: list[str] = []
        if "category" in df.columns:
            categories = sorted(df["category"].dropna().unique().tolist())
            selected_categories = st.multiselect("Category", categories, default=categories)

        selected_provinces: list[str] = []
        if "province" in df.columns:
            provinces = sorted(df["province"].dropna().unique().tolist())
            selected_provinces = st.multiselect("Province", provinces)

        selected_sentiments: list[str] = []
        if "sentiment" in df.columns:
            sentiments = sorted(df["sentiment"].dropna().unique().tolist())
            selected_sentiments = st.multiselect("Sentiment", sentiments)

        rating_range: tuple[float, float] | None = None
        if rating_column:
            ratings = pd.to_numeric(df[rating_column], errors="coerce").dropna()
            if not ratings.empty:
                min_rating = float(np.floor(ratings.min()))
                max_rating = float(np.ceil(ratings.max()))
                if min_rating < max_rating:
                    rating_range = st.slider(
                        rating_column.replace("_", " ").title(),
                        min_value=min_rating,
                        max_value=max_rating,
                        value=(min_rating, max_rating),
                        step=0.1,
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
    if rating_column and rating_range:
        ratings = pd.to_numeric(df[rating_column], errors="coerce")
        mask &= ratings.between(rating_range[0], rating_range[1])

    return mask


def render_overview(filtered_df: pd.DataFrame, manifest: dict, rating_column: str | None) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric(row_label(manifest), f"{len(filtered_df):,}")
    metric_cols[1].metric(
        "Unique Places",
        f"{filtered_df['place_name'].nunique():,}" if "place_name" in filtered_df.columns else "-",
    )
    metric_cols[2].metric(
        "Provinces",
        f"{filtered_df['province'].nunique():,}" if "province" in filtered_df.columns else "-",
    )
    metric_cols[3].metric(
        "Avg Rating",
        f"{pd.to_numeric(filtered_df[rating_column], errors='coerce').mean():.2f}"
        if rating_column
        else "-",
    )

    chart_cols = st.columns((1.1, 1))

    with chart_cols[0]:
        if rating_column:
            rating_counts = (
                pd.to_numeric(filtered_df[rating_column], errors="coerce")
                .round(1)
                .value_counts()
                .rename_axis("rating")
                .reset_index(name=row_label(manifest).lower())
                .sort_values("rating")
            )
            fig = px.bar(
                rating_counts,
                x="rating",
                y=row_label(manifest).lower(),
                color=row_label(manifest).lower(),
                color_continuous_scale=["#94a3b8", "#2563eb"],
                labels={"rating": "Rating"},
                title="Rating Distribution",
            )
            fig.update_layout(
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, width="stretch")

    with chart_cols[1]:
        if "sentiment" in filtered_df.columns:
            mix_column = "sentiment"
            title = "Sentiment Mix"
        elif "category" in filtered_df.columns:
            mix_column = "category"
            title = "Category Mix"
        else:
            mix_column = None
            title = ""

        if mix_column:
            mix_counts = (
                filtered_df[mix_column]
                .fillna("unknown")
                .value_counts()
                .rename_axis(mix_column)
                .reset_index(name=row_label(manifest).lower())
            )
            fig = px.pie(
                mix_counts,
                names=mix_column,
                values=row_label(manifest).lower(),
                hole=0.55,
                title=title,
                color_discrete_sequence=["#16a34a", "#64748b", "#dc2626", "#2563eb"],
            )
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, width="stretch")


def display_columns(df: pd.DataFrame, text_column: str, rating_column: str | None) -> list[str]:
    columns = [
        "similarity",
        "category",
        "place_name",
        "province",
        rating_column,
        "sentiment",
        "review_count",
        "sampled_review_count",
        "positive_review_count",
        "neutral_review_count",
        "negative_review_count",
        text_column,
    ]
    return [column for column in columns if column and column in df.columns]


def render_search(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    mask: pd.Series,
    manifest: dict,
    text_column: str,
    rating_column: str | None,
) -> None:
    st.subheader("Semantic Search")
    query = st.text_input(
        "Search by meaning",
        placeholder="เช่น ร้านอาหารติดทะเล บรรยากาศดี ราคาไม่แพง",
    )

    filtered_df = df[mask].copy()
    filtered_embeddings = embeddings[mask.to_numpy(dtype=bool)]

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    if not query.strip():
        preview_columns = [column for column in display_columns(filtered_df, text_column, rating_column) if column != "similarity"]
        st.dataframe(
            filtered_df[preview_columns].head(100),
            width="stretch",
            hide_index=True,
            column_config={
                text_column: st.column_config.TextColumn(text_column.replace("_", " ").title(), width="large"),
            },
        )
        return

    model_name = manifest_model_name(manifest)
    model = load_model(model_name)
    query_text = format_for_query(model_name, query.strip())
    query_embedding = model.encode([query_text], normalize_embeddings=True)[0]
    scores = filtered_embeddings @ query_embedding

    result_df = filtered_df.copy()
    result_df["similarity"] = scores
    result_df = result_df.sort_values("similarity", ascending=False).head(st.session_state["top_k"])

    columns = display_columns(result_df, text_column, rating_column)
    st.dataframe(
        result_df[columns],
        width="stretch",
        hide_index=True,
        column_config={
            "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
            text_column: st.column_config.TextColumn(text_column.replace("_", " ").title(), width="large"),
        },
    )

    if manifest_level(manifest) != "place" and {"place_name", "province", "category"}.issubset(result_df.columns):
        st.subheader("Top Matching Places")
        agg_map = {
            "avg_similarity": ("similarity", "mean"),
            "matched_rows": (text_column, "count"),
        }
        if rating_column:
            agg_map["avg_rating"] = (rating_column, "mean")

        place_scores = (
            result_df.groupby(["place_name", "province", "category"], as_index=False)
            .agg(**agg_map)
            .sort_values(["avg_similarity", "matched_rows"], ascending=[False, False])
        )
        st.dataframe(place_scores, width="stretch", hide_index=True)


def render_projection(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    manifest: dict,
    text_column: str,
    rating_column: str | None,
) -> None:
    if len(df) < 3:
        return

    st.subheader("Embedding Map")
    id_column = resolve_id_column(df)
    if id_column:
        row_ids = tuple(df[id_column].head(2500).astype(int))
    else:
        row_ids = tuple(range(min(len(df), 2500)))

    projection = build_projection(embeddings, row_ids)
    if projection.empty:
        return

    plot_df = df.head(len(projection)).copy()
    plot_df[["x", "y"]] = projection[["x", "y"]]
    plot_df["short_text"] = plot_df[text_column].astype(str).str.slice(0, 140)

    hover_data = {
        "x": False,
        "y": False,
        "short_text": True,
    }
    for column in ["place_name", "province", rating_column]:
        if column and column in plot_df.columns:
            hover_data[column] = True

    color = "sentiment" if "sentiment" in plot_df.columns else "category" if "category" in plot_df.columns else None
    symbol = "category" if color != "category" and "category" in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color=color,
        symbol=symbol,
        hover_data=hover_data,
        title=f"PCA Projection of {row_label(manifest)} Embeddings",
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
    manifests = load_manifests()
    manifest = choose_manifest(manifests)

    try:
        df, embeddings = load_embedding_data(
            manifest["metadata_file"],
            manifest["embeddings_file"],
        )
    except FileNotFoundError:
        render_missing_embeddings()

    model_name = manifest_model_name(manifest)
    text_column = resolve_text_column(df, manifest)
    rating_column = resolve_rating_column(df)

    st.title("Embedding Search Tester")
    st.caption(
        f"Model: {model_name} | Dataset: {manifest['_manifest_file']} | "
        f"Rows: {len(df):,} | Text column: {text_column}"
    )

    mask = filter_rows(df, manifest, rating_column)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        st.warning("No rows match the current filters.")
        return

    render_overview(filtered_df, manifest, rating_column)

    tab_search, tab_map, tab_data = st.tabs(["Search", "Embedding Map", "Data"])
    with tab_search:
        render_search(df, embeddings, mask, manifest, text_column, rating_column)
    with tab_map:
        render_projection(
            filtered_df.reset_index(drop=True),
            embeddings[mask.to_numpy(dtype=bool)],
            manifest,
            text_column,
            rating_column,
        )
    with tab_data:
        st.dataframe(filtered_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
