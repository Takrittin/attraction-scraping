# Places API Review Scraper and Embedding Dashboard

This project uses the Google Places API (New) to scrape, clean, and process attraction and restaurant review data. It also includes a place-summary embedding pipeline and a Streamlit dashboard for semantic place search.

The default data target is southern Thailand, but you can modify the province list and search terms to collect places from other regions.

## Features

- **Automated Scraping:** Fetches places and reviews for target regions (easily customizable in the script).
- **Two Categories:** Supports scraping both `attractions` and `restaurants`.
- **Review Categorization:** Splits reviews into `positive`, `neutral`, and `negative` CSV files based on star ratings.
- **Data Summarization:** Generates a clean summary file containing coordinates (lat/lng), average ratings, and review counts—perfect for mapping and distance calculations.
- **Local Cache:** Can re-process local raw data without making additional expensive API calls using the `--from-raw` flag.
- **Place Summary Embeddings:** Summarizes all reviews for each place into one short English description, then converts each place summary into a semantic vector.
- **Streamlit Dashboard:** Provides semantic search, filters, charts, matching-place summaries, and an embedding map.

## Customizing Locations

To scrape data for a different country, state, or region, simply open the Python script (`scrape_southern_thailand_attractions.py`) and modify the `SOUTHERN_PROVINCES` list near the top of the file with your desired areas:

```python
# Change these to your desired search areas!
SOUTHERN_PROVINCES = [
    Province("Tokyo", "โตเกียว"),
    Province("Osaka", "โอซาก้า"),
    # Add any city, state, or region here!
]
```

## Prerequisites

- Python 3.8+
- [Google Places API Key](https://console.cloud.google.com/apis) (with Places API (New) enabled)
- Required Python packages from `requirements.txt`, including `requests`, `pandas`, `streamlit`, `sentence-transformers`, `torch`, `scikit-learn`, `plotly`, and `pyarrow`.

```bash
pip install -r requirements.txt
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/Takrittin/attraction-scraping.git
   cd attraction-scraping
   ```

2. Create a `.env.local` file in the root directory and add your Google Places API Key:
   ```env
   GOOGLE_PLACES_API_KEY=your_api_key_here
   ```
   *(Note: `.env.local` is ignored by git to keep your credentials secure.)*

## Usage

### Full Workflow

1. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Scrape or process review data.

3. Generate English place summary embeddings.

4. Run the Streamlit dashboard.

5. Search places by meaning.

### 1. Scrape Attractions (Default)
Run the script without any arguments to scrape attractions. The data will be saved in the `attraction_output/` folder.
```bash
python scrape_southern_thailand_attractions.py
```

### 2. Scrape Restaurants
Use the `--category` flag to scrape restaurants. The data will be saved in the `restaurant_output/` folder.
```bash
python scrape_southern_thailand_attractions.py --category restaurants
```

### 3. Process Local Data (Offline Mode)
If you have already downloaded the raw data and want to re-run the cleaning and categorization process without calling the Google API again, use the `--from-raw` flag.
```bash
python scrape_southern_thailand_attractions.py --category attractions --from-raw
```

### Advanced Options
You can view all available CLI arguments by running:
```bash
python scrape_southern_thailand_attractions.py --help
```
- `--max-pages-per-province`: Set the maximum number of search result pages per province (Default is 3).
- `--language`: Set the preferred language for results (Default is `th`).
- `--output-dir`: Override the default output directory.

## English Place Summary Embeddings

If you want English-only place vectors, summarize all reviews for each place
into one English summary of 40 words or fewer, then embed those summaries with
`sentence-transformers/all-MiniLM-L6-v2`:

```bash
python create_english_review_embeddings.py
```

This command reads both cleaned review files, creates one
`english_40word_summary` row per place, and writes:

- `embeddings/place_metadata_english_40words.parquet`
- `embeddings/place_embeddings_english_40words.npy`
- `embeddings/manifest_place_english_40words.json`

The default summarization provider is Vertex AI with `gemini-2.5-flash-lite`.
This is recommended when you are using Google Cloud Platform instead of Google
AI Studio. It uses your active `gcloud` login and Google Cloud project billing,
not a Gemini API key from AI Studio.

First, make sure you are logged in to Google Cloud:

```bash
gcloud auth login
gcloud config set project your_google_cloud_project_id
```

Then add your project settings to `.env.local`:

```env
VERTEX_PROJECT_ID=your_google_cloud_project_id
VERTEX_LOCATION=us-central1
```

To test the pipeline on a few places before running the full dataset:

```bash
python create_english_review_embeddings.py --limit 20
```

You can also run the work in two separate stages.

Stage 1 creates only the place summaries and saves the metadata file:

```bash
python create_english_review_embeddings.py --skip-embedding --transform-batch-size 10
```

Stage 2 uses the saved summaries and creates only the vector files:

```bash
python create_english_review_embeddings.py --summary-provider existing --embedding-batch-size 128
```

Use the two-stage flow when you want to inspect or share the summaries before
creating vectors, or when you want to avoid calling Gemini again.

For a small two-stage test, keep the same `--limit` in both commands:

```bash
python create_english_review_embeddings.py --limit 20 --skip-embedding --transform-batch-size 5
python create_english_review_embeddings.py --limit 20 --summary-provider existing --embedding-batch-size 128
```

By default, each place summary uses up to 20 review texts from that place. You
can increase or decrease that sample size:

```bash
python create_english_review_embeddings.py --max-reviews-per-place 30
```

If you want to use the Google AI Studio Gemini API instead of Vertex AI, pass
`--summary-provider gemini` and add `GEMINI_API_KEY` to `.env.local`.

The script saves summary checkpoints as it runs. If it stops midway, run the
same command again and it will reuse already-created summaries.

If you prefer a local Hugging Face translation model instead of an LLM API, you
can run:

```bash
python create_english_review_embeddings.py --summary-provider transformers
```

That fallback translates Thai text with `Helsinki-NLP/opus-mt-th-en`, then trims
the translated text to 40 words. Gemini is recommended when you need better
summaries, because it translates and summarizes in one step.

## Streamlit Dashboard

Run the dashboard after generating place summary embeddings:

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard supports:

- Semantic place search
- Category, province, sentiment, and rating filters
- Top matching place summaries
- Rating and sentiment charts
- 2D embedding map

Example search queries:

```text
ร้านอาหารติดทะเล บรรยากาศดี ราคาไม่แพง
```

```text
สถานที่เงียบสงบ เหมาะกับครอบครัว
```

```text
บริการไม่ดี รอนาน
```

## Updating Embeddings After New Reviews

If new reviews are added, rerun the two-stage place summary pipeline:

```bash
python create_english_review_embeddings.py --skip-embedding --transform-batch-size 10
python create_english_review_embeddings.py --summary-provider existing --embedding-batch-size 128
```

The script reuses existing place summaries when possible. Use
`--force-transform` only when you want to regenerate every place summary.

## Output Structure

The project generates well-organized data ready for mapping, analysis, semantic search, and dashboard use.

```text
.
├── scrape_southern_thailand_attractions.py
├── create_english_review_embeddings.py
├── streamlit_app.py
├── attraction_output/ (or restaurant_output/)
│   ├── raw/
│   │   ├── raw_attractions.csv      # Untouched place details from the API
│   │   └── raw_reviews.csv          # Untouched review data from the API
│   └── cleaned/
│       ├── attraction_summary.csv   # Coordinates, ratings, and stats per place
│       ├── all_reviews.csv          # Combined clean reviews
│       ├── positive_reviews.csv     # Reviews with 4-5 stars
│       ├── neutral_reviews.csv      # Reviews with 3 stars
│       └── negative_reviews.csv     # Reviews with 1-2 stars
└── embeddings/
    ├── place_embeddings_english_40words.npy       # Place summary vectors
    ├── place_metadata_english_40words.parquet     # Place rows aligned with vectors
    └── manifest_place_english_40words.json        # Embedding metadata
```

For restaurants, the same structure is generated in `restaurant_output/`:

```text
restaurant_output/
├── raw/
│   ├── raw_restaurants.csv      # Untouched place details from the API
│   └── raw_reviews.csv          # Untouched review data from the API
└── cleaned/
    ├── restaurant_summary.csv   # Coordinates, ratings, and stats per place
    ├── all_reviews.csv          # Combined clean reviews
    ├── positive_reviews.csv     # Reviews with 4-5 stars
    ├── neutral_reviews.csv      # Reviews with 3 stars
    └── negative_reviews.csv     # Reviews with 1-2 stars
```
