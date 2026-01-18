"""
Scraper module for World Athletics Top Lists.
Contains classes for fetching and parsing data from World Athletics website.
"""

import requests
import time
import logging
from bs4 import BeautifulSoup
from urllib.parse import urlencode
import json  # To work with JSON data
import ssl
import urllib3

# Fix SSL issues on macOS
ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorldAthleticsScraper:
    """
    Scraper for World Athletics Top Lists website.
    """

    def __init__(self, base_url='https://worldathletics.org/records/toplists',
                 max_retries=3, request_delay=0.5):
        """
        Initialize the scraper.

        Args:
            base_url (str): Base URL for World Athletics top lists
            max_retries (int): Maximum number of retries for failed requests
            request_delay (float): Delay between requests in seconds
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.request_delay = request_delay
        self.session = requests.Session()
        self.graphql_url = "https://worldathletics.org/api/v1/graphql"  # Store GraphQL URL

    def construct_url(self, event_category, event, gender, age_category, year, **kwargs):
        """
        Construct URL for World Athletics top lists.

        Args:
            event_category (str): Category of event (sprints, jumps, etc.)
            event (str): Event name (100-metres, long-jump, etc.)
            gender (str): Gender (men, women)
            age_category (str): Age category (senior, u20, etc.)
            year (int): Year
            **kwargs: Additional query parameters

        Returns:
            str: Constructed URL
        """
        url = f'{self.base_url}/{event_category}/{event}/all/{gender}/{age_category}/{year}'
        if kwargs:
            url = f'{url}?{urlencode(kwargs)}'
        return url

    def fetch_page(self, url):
        """
        Fetch page content with retry mechanism.

        Args:
            url (str): URL to fetch

        Returns:
            str or None: Page content or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching: {url} (Attempt {attempt+1}/{self.max_retries})")
                response = self.session.get(url, verify=False)
                response.raise_for_status()
                time.sleep(self.request_delay)  # Add delay to avoid rate limiting
                return response.text
            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {url}: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
                    return None

    def parse_results(self, html_content, event, gender, determinants_dict=None, medal_calculator=None):
        """
        Parse results from HTML content.

        Args:
            html_content (str): HTML content
            event (str): Event name
            gender (str): Gender
            determinants_dict (dict, optional): Dictionary with determinants data
            medal_calculator (MedalCalculator, optional): Medal calculator instance

        Returns:
            list or None: List of parsed results or None if no results
        """
        if not html_content:
            return None

        soup = BeautifulSoup(html_content, 'html.parser')

        # Check if results exist
        if 'No results found' in html_content or not soup.find(class_='records-table'):
            logger.info("No results found")
            return None

        results = []
        records_table = soup.find(class_='records-table')
        if records_table is None:
            logger.warning("Records table not found")
            return None

        # Process each row in the table
        for tr in records_table.find_all('tr')[1:]:  # Skip header row
            tds = tr.find_all('td')
            if len(tds) < 10:
                continue

            # Extract data from row
            row = [td.text.strip() for td in tds]

            # Handle wind column if missing
            if len(row) == 9:
                row.insert(2, 'N/A')

            # Add competitor URL if available
            if tds[3].find('a'):
                competitor_url = 'https://worldathletics.org' + tds[3].find('a')['href']
                row.insert(4, competitor_url)
            else:
                row.insert(4, '')

            # Process mark value
            from utils import convert_to_seconds, calculate_age
            row[1] = str(convert_to_seconds(row[1], event))

            # Calculate age from DOB
            dob = row[5]
            age = calculate_age(dob)
            row.append(str(age))

            # Add event metadata
            row.append(event)
            row.append('all')  # Environment
            row.append(gender)

            # Add medal metrics if medal calculator is provided
            if medal_calculator:
                row = medal_calculator.calculate_medal_metrics(row, event, gender)
            # Add determinants if available and medal calculator not provided
            elif determinants_dict:
                det_values = determinants_dict.get((event, gender), {})
                for key, value in det_values.items():
                    row.append(value)

            results.append(row)

        logger.info(f"Parsed {len(results)} results")
        return results

    def scrape_event(self, event_category, event, gender, age_category, year,
                     page=1, region_type='world', best_results_only=True,
                     track_size='regular', determinants_dict=None, medal_calculator=None):
        """
        Scrape results for a specific event, gender, age category, and year.

        Args:
            event_category (str): Category of event
            event (str): Event name
            gender (str): Gender
            age_category (str): Age category
            year (int): Year
            page (int): Page number
            region_type (str): Region type
            best_results_only (bool): Whether to include only best results
            track_size (str): Track size
            determinants_dict (dict, optional): Dictionary with determinants data
            medal_calculator (MedalCalculator, optional): Medal calculator instance

        Returns:
            list or None: List of parsed results or None if no results
        """
        url = self.construct_url(
            event_category, event, gender, age_category, year,
            page=page, regionType=region_type,
            bestResultsOnly=best_results_only, oversizedTrack=track_size
        )

        html_content = self.fetch_page(url)
        if not html_content:
            return None

        return self.parse_results(html_content, event, gender, determinants_dict, medal_calculator)