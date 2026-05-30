import trafilatura
from src.config import ADMIN_KEYWORDS, FACULTY_KEYWORDS, STUDENT_KEYWORDS

def extract_and_tag(url: str, html_content: str) -> dict | None:
    clean_text = trafilatura.extract(html_content)
    if not clean_text:
        return None

    url_lower = url.lower()
    role = "public"
    
    if any(keyword in url_lower for keyword in ADMIN_KEYWORDS):
        role = "administrator"
    elif any(keyword in url_lower for keyword in FACULTY_KEYWORDS):
        role = "faculty"
    elif any(keyword in url_lower for keyword in STUDENT_KEYWORDS):
        role = "student"

    return {"url": url, "role": role, "content": clean_text}