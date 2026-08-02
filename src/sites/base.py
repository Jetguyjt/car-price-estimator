"""
The SiteAdapter contract — everything a listing site must provide so the
collection runner and enricher can drive it without knowing any
site-specific details.

An adapter is pure URL-building + HTML-parsing. It never touches Selenium or
the database; the orchestrator owns the browser and persistence. That keeps
adapters unit-testable against saved HTML fixtures (tests/fixtures/<site>/).

All parse output uses the canonical listing schema (store.LISTING_COLUMNS
minus bookkeeping): the same raw-field names the cleaning stage already
understands, plus vin / dealer_city / dealer_state / listing_type.
"""

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

# VINs never contain I, O, or Q.
VIN_RE = re.compile(r"\b([A-HJ-NPR-Z0-9]{17})\b")
# "Some Dealer Name  Austin, TX 78701" — city, two-letter state, zip5.
CITY_STATE_ZIP_RE = re.compile(r"([A-Za-z .'\-]{3,40}),\s*([A-Z]{2})\s*(\d{5})")


def clean_int(value) -> Optional[int]:
    if value is None:
        return None
    digits = re.sub(r"[^\d]", "", str(value))
    return int(digits) if digits else None


def clean_float(value) -> Optional[float]:
    if value is None:
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def get_text(node) -> Optional[str]:
    if node is None:
        return None
    text = node.get_text(" ", strip=True)
    return text or None


class SiteAdapter(ABC):
    """One listing site. Subclasses are stateless."""

    site_name: str = ""
    rate_limit_s: float = 3.0          # minimum pause between page loads
    supports_stock_types: tuple = ("used",)

    @abstractmethod
    def build_search_url(self, *, seed_zip: str, radius_miles: int,
                         band_min: Optional[int], band_max: Optional[int],
                         stock_type: str, page: int) -> str:
        """Results-page URL for one (zip seed, price band, stock type, page)."""

    @abstractmethod
    def parse_cards(self, page_html: str) -> List[Dict]:
        """Card rows from a results page, in the canonical schema.
        Must include listing_key; should include listing_url."""

    @abstractmethod
    def detail_url(self, row: Dict) -> Optional[str]:
        """URL of the full listing page for an already-collected row."""

    @abstractmethod
    def parse_detail(self, page_html: str) -> Dict:
        """Detail-page fields (canonical raw columns + vin/dealer_* when found)."""

    # ---- shared extraction helpers ----

    @staticmethod
    def extract_vin(text: str) -> Optional[str]:
        m = re.search(r"vin[:\s#]*([a-hj-npr-z0-9]{17})\b", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        return None

    @staticmethod
    def extract_dealer_location(text: str) -> Dict[str, str]:
        m = CITY_STATE_ZIP_RE.search(text)
        if not m:
            return {}
        return {"dealer_city": m.group(1).strip(), "dealer_state": m.group(2),
                "zip": m.group(3)}
