# Places API Attractions & Restaurants Scraper

This Python script uses the Google Places API (New) to scrape, clean, and process data for tourist attractions and restaurants across predefined regions. While initially configured for provinces in Thailand, you can easily modify it to search anywhere in the world. It automatically handles pagination, rate-limiting, and categorizes reviews for easy downstream analysis.

## Features

- **Automated Scraping:** Fetches places and reviews for target regions (easily customizable in the script).
- **Two Categories:** Supports scraping both `attractions` and `restaurants`.
- **Review Categorization:** Splits reviews into `positive`, `neutral`, and `negative` CSV files based on star ratings.
- **Data Summarization:** Generates a clean summary file containing coordinates (lat/lng), average ratings, and review counts—perfect for mapping and distance calculations.
- **Local Cache:** Can re-process local raw data without making additional expensive API calls using the `--from-raw` flag.

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
- [Google Places API Key](https://developers.google.com/maps/documentation/places/web-service/get-api-key) (with Places API (New) enabled)
- Required Python packages: `requests`, `pandas`

```bash
pip install requests pandas
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

## Output Structure

The script generates well-organized data ready for mapping, analysis, and UI integration.

```text
attraction_output/ (or restaurant_output/)
├── raw/
│   ├── raw_attractions.csv      # Untouched place details from the API
│   └── raw_reviews.csv          # Untouched review data from the API
└── cleaned/
    ├── attraction_summary.csv   # Coordinates, ratings, and stats per place
    ├── all_reviews.csv          # Combined clean reviews
    ├── positive_reviews.csv     # Reviews with 4-5 stars
    ├── neutral_reviews.csv      # Reviews with 3 stars
    └── negative_reviews.csv     # Reviews with 1-2 stars
```
