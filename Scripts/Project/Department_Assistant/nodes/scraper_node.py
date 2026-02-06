"""
Scraper Node - Web Scraping Component
This node crawls the department website and extracts text content
"""
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import Set, List, Dict, Tuple
import sys
import os
from pathlib import Path

# Add parent directory to path to import config and state
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_URL, ALLOWED_DOMAIN, SCRAPER_DELAY, MAX_PAGES, REQUEST_TIMEOUT
from state import PipelineState

# Create directory for downloaded PDFs (using absolute path)
PDF_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "downloaded_pdfs")
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)


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
    
    # Continue until queue is empty or we hit max pages
    while to_visit and len(visited) < MAX_PAGES:
        # Get next URL from queue
        current_url = to_visit.pop(0)
        
        # Skip if already visited
        if current_url in visited:
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
    
    Args:
        url: URL of the page to scrape
        
    Returns:
        Dictionary with 'text' and 'links' keys
    """
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
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Error fetching {url}: {e}")
        return None


def extract_text(soup: BeautifulSoup) -> str:
    """
    Extract clean text from HTML soup
    Removes scripts, styles, but keeps important content areas
    
    Args:
        soup: BeautifulSoup object
        
    Returns:
        Cleaned text content
    """
    # Remove only truly unwanted elements (scripts, styles, etc.)
    # Keep nav, header as they might contain important info on homepage
    for element in soup(['script', 'style', 'noscript']):
        element.decompose()  # Remove from tree
    
    # Try to get main content area (common patterns)
    # This focuses on the actual content, not navigation/sidebars
    main_content = (
        soup.find('main') or           # HTML5 main tag
        soup.find('article') or        # Article tag
        soup.find('div', class_='content') or  # Common class name
        soup.find('div', id='content') or
        soup.find('div', role='main') or  # ARIA role
        soup.body                      # Fallback to body (includes more content)
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
    
    Args:
        url: URL of the PDF file
        
    Returns:
        Path to downloaded file, or None if download failed
    """
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
        print(f"⚠️  Error downloading PDF {url}: {e}")
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

