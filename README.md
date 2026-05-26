# Places API Review Scraper and Embedding Dashboard

This project uses the Google Places API (New) to scrape, clean, and process attraction and restaurant review data. It also includes a sentence embedding pipeline and a Streamlit dashboard for semantic review search.

The default data target is southern Thailand, but you can modify the province list and search terms to collect places from other regions.

## Features

- **Automated Scraping:** Fetches places and reviews for target regions (easily customizable in the script).
- **Two Categories:** Supports scraping both `attractions` and `restaurants`.
- **Review Categorization:** Splits reviews into `positive`, `neutral`, and `negative` CSV files based on star ratings.
- **Data Summarization:** Generates a clean summary file containing coordinates (lat/lng), average ratings, and review counts—perfect for mapping and distance calculations.
- **Local Cache:** Can re-process local raw data without making additional expensive API calls using the `--from-raw` flag.
- **Sentence Embeddings:** Converts review text into semantic vectors using multilingual models such as `BAAI/bge-m3`.
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

3. Generate sentence embeddings.

4. Run the Streamlit dashboard.

5. Search reviews by meaning.

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

## Sentence Embeddings

Generate sentence embeddings from cleaned reviews:

```bash
python create_review_embeddings.py --model BAAI/bge-m3 --batch-size 16
```

The script reads:

- `restaurant_output/cleaned/all_reviews.csv`
- `attraction_output/cleaned/all_reviews.csv`

It creates:

- `embeddings/review_embeddings.npy`
- `embeddings/review_metadata.parquet`
- `embeddings/manifest.json`

`BAAI/bge-m3` is recommended for better multilingual Thai/English search quality. If you want a faster lightweight test model, you can use:

```bash
python create_review_embeddings.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --batch-size 64
```

## Streamlit Dashboard

Run the dashboard after generating embeddings:

```bash
streamlit run streamlit_app.py --server.fileWatcherType none
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

The dashboard supports:

- Semantic review search
- Category, province, sentiment, and rating filters
- Top matching review results
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

If new reviews are added, run the embedding command again:

```bash
python create_review_embeddings.py --model BAAI/bge-m3 --batch-size 16
```

The current script regenerates all embeddings. This is simple and safe for the current dataset size. For a much larger dataset, you can add an incremental update script later to embed only new reviews and append them to the saved vector files.

## Output Structure

The project generates well-organized data ready for mapping, analysis, semantic search, and dashboard use.

```text
.
├── scrape_southern_thailand_attractions.py
├── create_review_embeddings.py
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
    ├── review_embeddings.npy        # Sentence embedding matrix
    ├── review_metadata.parquet      # Review rows aligned with the vectors
    └── manifest.json                # Model name and embedding metadata
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
