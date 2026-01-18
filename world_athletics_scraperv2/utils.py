"""
Utility module for World Athletics Top Lists Scraper.
Contains helper functions for data processing and conversion.
"""

from datetime import datetime
import re


def convert_to_seconds(time_str, event):
    """
    Convert time string to seconds.

    Args:
        time_str (str): Time string (MM:SS.ms, HH:MM:SS)
        event (str): Event name

    Returns:
        float or str: Time in seconds or original string if conversion fails
    """
    try:
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = map(float, parts)
                return hours * 3600 + minutes * 60 + seconds
            elif len(parts) == 2:  # MM:SS.ms
                minutes, seconds = map(float, parts)
                return minutes * 60 + seconds
        return float(time_str)
    except ValueError:
        return time_str  # Return original if conversion fails


def calculate_age(dob):
    """
    Calculate age from date of birth.

    Args:
        dob (str): Date of birth (DD MMM YYYY)

    Returns:
        int or None: Age or None if fails
    """
    try:
        dob_obj = datetime.strptime(dob, '%d %b %Y')
        today = datetime.today()
        return today.year - dob_obj.year - ((today.month, today.day) < (dob_obj.month, dob_obj.day))
    except ValueError:
        return None  # Return None for missing/invalid DOB


def clean_mark(mark):
    """
    Clean mark by removing non-numeric chars.

    Args:
        mark (str): Mark value

    Returns:
        str: Cleaned mark
    """
    return re.sub(r'[^0-9.]', '', str(mark)) if mark else ''  # Handle None/empty


def create_directory(directory):
    """Create directory if not exists."""
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)