"""
Rebuild script - wipes and re-scrapes the department knowledge base from scratch.

What it does:
1. Backs up the current department_vector_db/ and downloaded_pdfs/ to timestamped
   folders (so you can roll back if the new scrape turns out worse than what you have).
2. Runs the full data collection workflow (scraper -> processor) via main.collect_data():
   scrapes che.iitb.ac.in fresh (up to MAX_PAGES from config.py), downloads PDFs,
   chunks everything, generates embeddings, and stores it all in a brand-new
   department_vector_db/.

Run this with your project venv from inside the Department_Assistant folder, e.g.:
    & "..\\..\\..\\venv\\Scripts\\python.exe" rebuild.py

Expect this to take a while (well over half an hour with MAX_PAGES=350 and the
1.5s per-request politeness delay, plus PDF downloads and embedding generation) -
that's expected. It's the same collect_data() pipeline the Streamlit "Initialize
Database" button calls, just run standalone so a 30+ minute request doesn't sit
inside (and risk timing out) a Streamlit session.
"""
import os
import shutil
from datetime import datetime

from config import VECTOR_DB_PATH, BASE_DIR

PDF_DIR = str(BASE_DIR / "downloaded_pdfs")


def backup_if_exists(path: str) -> None:
    if os.path.exists(path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}_backup_{timestamp}"
        print(f"Backing up {path} -> {backup_path}")
        shutil.move(path, backup_path)
    else:
        print(f"Nothing to back up at {path} (doesn't exist yet)")


def main() -> None:
    print("=" * 80)
    print("REBUILDING DEPARTMENT ASSISTANT KNOWLEDGE BASE")
    print("=" * 80)

    backup_if_exists(VECTOR_DB_PATH)
    backup_if_exists(PDF_DIR)

    # Imported after the backup so we're not holding any handle on the old DB
    from main import collect_data

    print("\nStarting fresh scrape + process. This will take a while - grab a coffee...\n")
    result = collect_data()

    print("\n" + "=" * 80)
    if result.get("error"):
        print(f"FAILED: {result['error']}")
    else:
        print("DONE")
        print(f"Status: {result.get('status')}")
        print(f"Pages scraped: {len(result.get('scraped_pages') or [])}")
        print(f"PDFs downloaded: {len(result.get('pdf_files') or [])}")
        print(f"Chunks stored: {len(result.get('stored_doc_ids') or [])}")
    print("=" * 80)


if __name__ == "__main__":
    main()
