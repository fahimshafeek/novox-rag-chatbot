from bs4 import BeautifulSoup
from src.config import ADMIN_KEYWORDS, FACULTY_KEYWORDS, STUDENT_KEYWORDS

def extract_and_tag(url: str, soup: BeautifulSoup) -> dict | None:
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Get title
    title = soup.title.string if soup.title else ""
    
    # Get text content
    text = soup.get_text(separator=' ', strip=True)
    
    if not text:
        return None

    content = f"{title}\n\n{text}" if title else text

    url_lower = url.lower()
    role = "public"
    
    if any(keyword in url_lower for keyword in ADMIN_KEYWORDS):
        role = "administrator"
    elif any(keyword in url_lower for keyword in FACULTY_KEYWORDS):
        role = "faculty"
    elif any(keyword in url_lower for keyword in STUDENT_KEYWORDS):
        role = "student"

    return {"url": url, "role": role, "content": content}