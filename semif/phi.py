"""Scan text for patterns that look like personal data. The demo data must never match any of them."""
import re

PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?!\d)"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b"),
    "url": re.compile(r"\bhttps?://|\bwww\.", re.I),
}


def scan(text: str) -> list[str]:
    """Names of the patterns found (never the matched text)."""
    return [f"looks like {name}" for name, p in PATTERNS.items() if p.search(text)]
