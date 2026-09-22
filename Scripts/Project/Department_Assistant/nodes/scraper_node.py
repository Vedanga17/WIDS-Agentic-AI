"""
Scraper Node - Web Scraping Component
This node crawls the department website and extracts text content
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
from typing import Set, List, Dict, Tuple
import sys
import os
from pathlib import Path

# Add parent directory to path to import config and state
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_URL, ALLOWED_DOMAIN, SCRAPER_DELAY, MAX_PAGES, REQUEST_TIMEOUT, SCRAPER_MAX_RETRIES
from state import PipelineState

# Create directory for downloaded PDFs (using absolute path)
PDF_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloaded_pdfs")
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)

# Tags that are pure site-chrome (menus, footers) rather than page content.
# Stripped out before extraction so the same nav/footer text doesn't get duplicated
# into every single page's chunks and drown out the actual content during retrieval.
BOILERPLATE_TAGS = ['nav', 'header', 'footer']


def scraper_node(state: PipelineState) -> PipelineState:
    """
    Main scraper node function
    Crawls the department website and extracts text content

    Args:
        state: Current pipeline state

    Returns:
        Updated state with scraped_pages populated
    """
    print("🕷️  Starting web scraper...")

    try:
        # Run the scraper
        scraped_data, pdf_files = scrape_website(BASE_URL)

        # Update state with scraped data
        state["scraped_pages"] = scraped_data
        state["pdf_files"] = pdf_files
        state["status"] = f"Successfully scraped {len(scraped_data)} pages and {len(pdf_files)} PDFs"

        print(f"✅ Scraping complete! Collected {len(scraped_data)} pages and {len(pdf_files)} PDFs")

    except Exception as e:
        state["error"] = f"Scraping failed: {str(e)}"
        state["status"] = "Failed"
        print(f"❌ Scraping error: {str(e)}")

    return state


def load_robot_parser(start_url: str) -> RobotFileParser:
    """
    Fetch and parse robots.txt for the site once, so the crawler can skip
    paths the site doesn't want crawled.

    Returns a RobotFileParser. If robots.txt can't be fetched/parsed for any
    reason, returns a permissive parser (fails open) rather than blocking the
    whole crawl.
    """
    parsed = urlparse(start_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        print(f"🤖 Loaded robots.txt from {robots_url}")
    except Exception as e:
        print(f"⚠️  Could not load robots.txt ({e}) - proceeding without robots restrictions")
        rp = None
    return rp


def scrape_website(start_url: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Crawl the website starting from start_url
    Uses BFS (Breadth-First Search) to discover pages

    Args:
        start_url: URL to start crawling from

    Returns:
        Tuple of (scraped_pages, pdf_files)
    """
    # Data structures for tracking
    visited: Set[str] = set()           # URLs we've already scraped
    to_visit: List[str] = [start_url]   # Queue of URLs to scrape
    scraped_pages: List[Dict[str, str]] = []  # Web pages
    pdf_files: List[Dict[str, str]] = []      # Downloaded PDFs

    robot_parser = load_robot_parser(start_url)

    # Continue until queue is empty or we hit max pages
    while to_visit and len(visited) < MAX_PAGES:
        # Get next URL from queue
        current_url = to_visit.pop(0)

        # Skip if already visited
        if current_url in visited:
            continue

        # Respect robots.txt, if we managed to load one
        if robot_parser is not None and not robot_parser.can_fetch("*", current_url):
            print(f"🚫 Skipping (robots.txt disallows): {current_url}")
            visited.add(current_url)
            continue

        # Check if it's a PDF
        if current_url.lower().endswith('.pdf'):
            # Download PDF
            pdf_path = download_pdf(current_url)
            if pdf_path:
                pdf_files.append({
                    "url": current_url,
                    "file_path": pdf_path
                })
                print(f"📥 Downloaded PDF: {current_url}")
        else:
            # Scrape the web page
            page_content = scrape_single_page(current_url)

            if page_content:
                # Save the content
                scraped_pages.append({
                    "url": current_url,
                    "content": page_content["text"],
                    "type": "web"
                })

                # Progress indicator
                print(f"✓ Scraped {len(visited) + 1}/{MAX_PAGES}: {current_url}")

                # Find new links on this page
                new_links = page_content["links"]

                # Filter and add new links to queue
                for link in new_links:
                    # Only add if not visited and within allowed domain
                    if link not in visited and is_valid_url(link):
                        to_visit.append(link)

        # Mark as visited
        visited.add(current_url)

        # Be polite - wait before next request
        time.sleep(SCRAPER_DELAY)

    return scraped_pages, pdf_files


def scrape_single_page(url: str) -> Dict:
    """
    Scrape a single page and extract text content and links
    Retries transient failures (timeouts, connection errors, 5xx) a
    few times before giving up - permanent failures (403/404) are not retried.

    Args:
        url: URL of the page to scrape

    Returns:
        Dictionary with 'text' and 'links' keys
    """
    last_error = None
    for attempt in range(SCRAPER_MAX_RETRIES + 1):
        try:
            # Send HTTP GET request
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()  # Raise error for bad status codes

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')

            # Extract text content
            text_content = extract_text(soup)

            # Extract links
            links = extract_links(soup, url)

            return {
                "text": text_content,
                "links": links
            }

        except requests.exceptions.HTTPError as e:
            # 4xx/5xx - only retry server errors (5xx), not client errors like 403/404
            status = e.response.status_code if e.response is not None else None
            if status and 500 <= status < 600 and attempt < SCRAPER_MAX_RETRIES:
                print(f"⚠️  Server error {status} on {url}, retrying ({attempt + 1}/{SCRAPER_MAX_RETRIES})...")
                time.sleep(SCRAPER_DELAY)
                last_error = e
                continue
            print(f"⚠️  Error fetching {url}: {e}")
            return None

        except requests.exceptions.RequestException as e:
            # Timeouts, connection errors, etc. - worth a retry
            if attempt < SCRAPER_MAX_RETRIES:
                print(f"⚠️  {type(e).__name__} on {url}, retrying ({attempt + 1}/{SCRAPER_MAX_RETRIES})...")
                time.sleep(SCRAPER_DELAY)
                last_error = e
                continue
            print(f"⚠️  Error fetching {url}: {e}")
            return None

    print(f"⚠️  Giving up on {url} after {SCRAPER_MAX_RETRIES} retries: {last_error}")
    return None


def extract_text(soup: BeautifulSoup) -> str:
    """
    Extract clean text from HTML soup
    Removes scripts/styles and site-chrome (nav/header/footer), keeps actual content

    Args:
        soup: BeautifulSoup object

    Returns:
        Cleaned text content
    """
    # Remove unwanted elements: scripts/styles, plus nav/header/footer boilerplate.
    # The same nav menu and footer links appear on every page of the site - keeping
    # them meant that identical chrome text got embedded into hundreds of chunks,
    # which drowned out the actual page-specific content during retrieval.
    for element in soup(['script', 'style', 'noscript'] + BOILERPLATE_TAGS):
        element.decompose()  # Remove from tree

    # Try to get main content area (common patterns)
    # This focuses on the actual content, not navigation/sidebars
    main_content = (
        soup.find('main') or           # HTML5 main tag
        soup.find('article') or        # Article tag
        soup.find('div', class_='content') or  # Common class name
        soup.find('div', id='content') or
        soup.find('div', role='main') or  # ARIA role
        soup.body                      # Fallback to body (nav/header/footer already stripped above)
    )

    if main_content:
        # Get text and clean it up
        text = main_content.get_text(separator=' ', strip=True)

        # Clean up whitespace but preserve meaningful line breaks
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = ' '.join(lines)

        # Final cleanup of multiple spaces
        text = ' '.join(text.split())

        return text

    return ""


def extract_links(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract all valid links from the page

    Args:
        soup: BeautifulSoup object
        base_url: Base URL for resolving relative links

    Returns:
        List of absolute URLs
    """
    links = []

    # Find all <a> tags with href attribute
    for anchor in soup.find_all('a', href=True):
        # Get the href value
        href = anchor.get('href')

        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, href)

        # Remove fragment identifiers (#section)
        absolute_url = absolute_url.split('#')[0]

        # Add to list if not already there
        if absolute_url and absolute_url not in links:
            links.append(absolute_url)

    return links


def download_pdf(url: str) -> str:
    """
    Download a PDF file from URL
    Retries transient failures a few times before giving up.

    Args:
        url: URL of the PDF file

    Returns:
        Path to downloaded file, or None if download failed
    """
    last_error = None
    for attempt in range(SCRAPER_MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()

            # Generate filename from URL
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'

            # Save to file
            file_path = os.path.join(PDF_DOWNLOAD_DIR, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)

            return file_path

        except Exception as e:
            if attempt < SCRAPER_MAX_RETRIES:
                print(f"⚠️  Error downloading PDF {url} ({e}), retrying ({attempt + 1}/{SCRAPER_MAX_RETRIES})...")
                time.sleep(SCRAPER_DELAY)
                last_error = e
                continue
            print(f"⚠️  Error downloading PDF {url}: {e}")
            return None

    print(f"⚠️  Giving up on PDF {url} after {SCRAPER_MAX_RETRIES} retries: {last_error}")
    return None


def is_valid_url(url: str) -> bool:
    """
    Check if URL should be scraped

    Filters out:
    - External domains
    - Non-HTML/PDF pages (images, etc.)
    - Login/logout pages
    - Fragment-only URLs

    Args:
        url: URL to validate

    Returns:
        True if URL should be scraped
    """
    parsed = urlparse(url)

    # Must be in allowed domain
    if ALLOWED_DOMAIN not in parsed.netloc:
        return False

    # Allow PDFs
    if url.lower().endswith('.pdf'):
        return True

    # Skip common file extensions
    skip_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.zip', '.doc', '.docx']
    if any(url.lower().endswith(ext) for ext in skip_extensions):
        return False

    # Skip login/logout pages
    skip_paths = ['/user/login', '/user/logout', '/admin']
    if any(path in url.lower() for path in skip_paths):
        return False

    # Skip fragment-only URLs (e.g., #section)
    if parsed.fragment and not parsed.path:
        return False

    return True


# For testing the scraper independently
if __name__ == "__main__":
    print("Testing scraper node...")

    # Create initial state
    test_state: PipelineState = {
        "scraped_pages": None,
        "pdf_files": None,
        "chunks": None,
        "stored_doc_ids": None,
        "query": None,
        "retrieved_docs": None,
        "response": None,
        "status": None,
        "error": None
    }

    # Run scraper
    result_state = scraper_node(test_state)

    # Print results
    if result_state.get("scraped_pages"):
        print(f"\n✅ Scraping successful!")
        print(f"Total pages: {len(result_state['scraped_pages'])}")
        print(f"Total PDFs: {len(result_state['pdf_files'])}")
        print(f"\nFirst page preview:")
        first_page = result_state["scraped_pages"][0]
        print(f"URL: {first_page['url']}")
        print(f"Content (first 500 chars): {first_page['content'][:500]}...")
    else:
        print(f"\n❌ Scraping failed: {result_state.get('error')}")
