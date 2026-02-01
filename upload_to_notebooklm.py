"""
Upload Briefings and Documents to NotebookLM.

This script uploads generated briefings and PDF documents to NotebookLM
for use as a RAG backend. Supports both the Python API and CLI fallback.

Prerequisites:
    pip install notebooklm-py
    notebooklm login  # One-time authentication

Usage:
    python upload_to_notebooklm.py                    # Upload all briefings
    python upload_to_notebooklm.py --docs             # Upload PDFs from documents/
    python upload_to_notebooklm.py --all              # Upload everything
    python upload_to_notebooklm.py --create-notebook  # Create notebook first
"""

import os
import subprocess
import sys
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
BRIEFINGS_DIR = BASE_DIR / "briefings"
DOCUMENTS_DIR = BASE_DIR / "documents"

# NotebookLM notebook name
NOTEBOOK_NAME = "KSA Athletics Intelligence"


def check_notebooklm_installed() -> bool:
    """Check if notebooklm-py is installed."""
    try:
        import notebooklm
        return True
    except ImportError:
        return False


def check_cli_available() -> bool:
    """Check if notebooklm CLI is available."""
    try:
        result = subprocess.run(
            ["notebooklm", "--version"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def create_notebook(name: str) -> bool:
    """Create a new NotebookLM notebook."""
    print(f"Creating notebook: {name}")

    try:
        # Try Python API first
        from notebooklm import NotebookLM
        nlm = NotebookLM()
        nlm.create(name)
        print(f"  Notebook '{name}' created successfully!")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"  API error: {e}")

    # Fallback to CLI
    if check_cli_available():
        result = subprocess.run(
            ["notebooklm", "create", name],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  Notebook '{name}' created successfully!")
            return True
        else:
            print(f"  CLI error: {result.stderr}")

    return False


def upload_file(filepath: Path, notebook: str) -> bool:
    """Upload a single file to NotebookLM."""
    print(f"  Uploading: {filepath.name}")

    try:
        # Try Python API first
        from notebooklm import NotebookLM
        nlm = NotebookLM()
        nlm.source_add(str(filepath), notebook=notebook)
        print(f"    Success!")
        return True
    except ImportError:
        pass
    except Exception as e:
        print(f"    API error: {e}")

    # Fallback to CLI
    if check_cli_available():
        result = subprocess.run(
            ["notebooklm", "source", "add", str(filepath), "--notebook", notebook],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"    Success!")
            return True
        else:
            print(f"    CLI error: {result.stderr}")

    return False


def upload_briefings(notebook: str = NOTEBOOK_NAME) -> int:
    """Upload all briefings from briefings/ directory."""
    if not BRIEFINGS_DIR.exists():
        print(f"Briefings directory not found: {BRIEFINGS_DIR}")
        print("Run 'python generate_briefings.py' first.")
        return 0

    print(f"\nUploading briefings to '{notebook}'...")

    success_count = 0
    for file in sorted(BRIEFINGS_DIR.glob("*.md")):
        if upload_file(file, notebook):
            success_count += 1

    print(f"\nUploaded {success_count} briefings.")
    return success_count


def upload_documents(notebook: str = NOTEBOOK_NAME) -> int:
    """Upload PDF documents from documents/ directory."""
    if not DOCUMENTS_DIR.exists():
        print(f"Documents directory not found: {DOCUMENTS_DIR}")
        return 0

    print(f"\nUploading documents to '{notebook}'...")

    success_count = 0
    for file in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        if upload_file(file, notebook):
            success_count += 1

    print(f"\nUploaded {success_count} documents.")
    return success_count


def show_setup_instructions():
    """Show setup instructions for NotebookLM."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    NotebookLM Setup Required                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. Install notebooklm-py:                                       ║
║     pip install notebooklm-py                                    ║
║                                                                   ║
║  2. Authenticate with Google:                                    ║
║     notebooklm login                                             ║
║     (Opens browser for Google sign-in)                           ║
║                                                                   ║
║  3. Run this script again:                                       ║
║     python upload_to_notebooklm.py --all                         ║
║                                                                   ║
║  Note: You need a free Google account.                           ║
║        Visit https://notebooklm.google.com to get started.       ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


def main():
    """Main entry point."""
    args = sys.argv[1:]

    # Check installation
    if not check_notebooklm_installed() and not check_cli_available():
        show_setup_instructions()
        return 1

    # Parse arguments
    create_nb = "--create-notebook" in args or "--create" in args
    upload_docs = "--docs" in args or "--documents" in args
    upload_all = "--all" in args

    # Create notebook if requested
    if create_nb:
        create_notebook(NOTEBOOK_NAME)

    # Upload briefings (default action)
    if upload_all or (not upload_docs):
        upload_briefings(NOTEBOOK_NAME)

    # Upload documents
    if upload_all or upload_docs:
        upload_documents(NOTEBOOK_NAME)

    print("\nDone!")
    print(f"\nView your notebook at: https://notebooklm.google.com")
    return 0


if __name__ == "__main__":
    sys.exit(main())
