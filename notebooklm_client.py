"""
NotebookLM Client - Query NotebookLM from Streamlit dashboard.

This module provides a simple interface to query your NotebookLM notebook
from the AI chatbot tab in the dashboard.

Usage:
    from notebooklm_client import query_notebook, NOTEBOOK_ID

    response = query_notebook("Who are the top KSA athletes?")
    print(response)
"""

import subprocess
import json
from typing import Optional

# Your NotebookLM notebook ID
NOTEBOOK_ID = "d7034cab-0282-4b95-b960-d8f5e40d90e1"


def check_notebooklm_available() -> bool:
    """Check if notebooklm CLI is available and authenticated."""
    try:
        result = subprocess.run(
            ["notebooklm", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def query_notebook(question: str, notebook_id: str = NOTEBOOK_ID) -> Optional[str]:
    """
    Query NotebookLM notebook and return the response.

    Args:
        question: The question to ask
        notebook_id: NotebookLM notebook ID (default: KSA Athletics)

    Returns:
        Response text from NotebookLM, or None if failed
    """
    try:
        result = subprocess.run(
            ["notebooklm", "ask", question, "--notebook", notebook_id],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"NotebookLM Error: {result.stderr.strip()}"

    except FileNotFoundError:
        return "NotebookLM CLI not installed. Run: pip install notebooklm-py"
    except subprocess.TimeoutExpired:
        return "NotebookLM query timed out. Try a simpler question."
    except Exception as e:
        return f"NotebookLM Error: {str(e)}"


def get_notebook_sources(notebook_id: str = NOTEBOOK_ID) -> list:
    """Get list of sources in the notebook."""
    try:
        result = subprocess.run(
            ["notebooklm", "source", "list", "--notebook", notebook_id],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            # Parse the output (table format)
            lines = result.stdout.strip().split('\n')
            sources = []
            for line in lines:
                if '|' in line and 'ID' not in line and '---' not in line:
                    parts = [p.strip() for p in line.split('|') if p.strip()]
                    if len(parts) >= 2:
                        sources.append({'id': parts[0], 'title': parts[1]})
            return sources
        return []
    except Exception:
        return []


# Quick test
if __name__ == "__main__":
    print("Testing NotebookLM connection...")

    if check_notebooklm_available():
        print("NotebookLM CLI is available")

        print("\nSources in notebook:")
        sources = get_notebook_sources()
        for s in sources:
            print(f"  - {s.get('title', 'Unknown')}")

        print("\nTesting query...")
        response = query_notebook("Who are the top 3 KSA athletes by world ranking?")
        print(f"\nResponse:\n{response}")
    else:
        print("NotebookLM CLI not available. Install with: pip install notebooklm-py")
