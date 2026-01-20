"""
Configuration module for World Athletics Top Lists Scraper.
Contains constants and settings for the scraper.
"""

# Event categories mapping
EVENTS = {
    '100-metres': 'sprints',
    '200-metres': 'sprints',
    '400-metres': 'sprints',
    '800-metres': 'middlelong',
    '1500-metres': 'middlelong',
    '3000-metres-steeplechase': 'middlelong',
    '5000-metres': 'middlelong',
    '10000-metres': 'middlelong',
    'pole-vault': 'jumps',
    'hammer-throw': 'throws',
    'discus-throw': 'throws',
    'long-jump': 'jumps',
    'triple-jump': 'jumps',
    'high-jump': 'jumps',
    'shot-put': 'throws',
    'javelin-throw': 'throws',
    'decathlon': 'combined-events',
    'heptathlon': 'combined-events',
    '3000-metres': 'endurance',
    '20-kilometres-walk': 'endurance',
    '50-kilometres-walk': 'endurance',
    'marathon': 'road-running',
    '400-metres-hurdles': 'hurdles',
    '110-metres-hurdles': 'hurdles',
    '100-metres-hurdles': 'hurdles',
    '400-metres-short-track': 'sprints',
    '800-metres-short-track': 'middlelong',
    '1500-metres-short-track': 'middlelong',
    '5000-metres-short-track': 'middlelong'
}

# Default scraping parameters
DEFAULT_YEARS = [2024, 2025]

# Historical years to fill the 2003-2019 gap
HISTORICAL_YEARS = list(range(2003, 2020))  # 2003-2019 inclusive
DEFAULT_NUM_PERFORMANCES = 7000
DEFAULT_AGE_CATEGORY = ['senior'] #U18, U20
DEFAULT_BEST_RESULTS_ONLY = [True]
DEFAULT_TRACK_SIZE = ['regular']
DEFAULT_REGION_TYPE = ['world']
DEFAULT_GENDER = ['men', 'women']

# Calculate number of pages based on performances per page (100)
def calculate_pages(num_performances):
    """Calculate number of pages to scrape based on total performances."""
    return range(1, int(num_performances / 100) + 1)

# CSV headers
HEADERS = [
    'Rank', 'Mark', 'Wind', 'Competitor', 'CompetitorURL',
    'DOB', 'Nat', 'Pos', '', 'Venue', 'Date', 'ResultScore',
    'Age', 'Event', 'Environment', 'Gender'
]

# Base URL for World Athletics top lists
BASE_URL = 'https://worldathletics.org/records/toplists'

# Maximum number of retries for failed requests
MAX_RETRIES = 3

# Delay between requests (in seconds) to avoid rate limiting
REQUEST_DELAY = 0.5

# Maximum number of concurrent requests
MAX_CONCURRENT_REQUESTS = 10



# Output directory for CSV files

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")


# Final database filenames
DB_FILENAME = 'db.csv'
DB_CLEANED_FILENAME = 'db_cleaned.csv'

