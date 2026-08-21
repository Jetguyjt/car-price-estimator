"""
cars.com adapter - port of the proven parsers from the earlier listing collector and
the earlier detail-page collector into the SiteAdapter contract, with two upgrades:

- searches are seeded per-ZIP3 (zip + radius params) instead of the old
  hardcoded nationwide zip=98103 / maximum_distance=9999;
- detail parsing now captures VIN and dealer city/state/zip, which the old
  pass fetched but discarded.
"""

import html
import json
import re
from typing import Dict, List, Optional
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup

from sites.base import SiteAdapter, clean_float, clean_int, get_text

SITE_ROOT = "https://www.cars.com"
BASE_RESULTS_URL = f"{SITE_ROOT}/shopping/results/"

LISTING_SELECTOR = "[data-listing-id]"

# Detail-page "Basics" dt/dd labels (lowercased) -> canonical raw columns.
LABEL_MAP = {
    "exterior color": "exterior_color",
    "interior color": "interior_color",
    "drivetrain": "drivetrain_raw",
    "fuel type": "fuel_type_raw",
    "transmission": "transmission_raw",
    "engine": "engine_raw",
    "mpg": "mpg_text",
    "stock #": "stock_number",
    "stock number": "stock_number",
}


def _parse_json_attr(raw_attr: Optional[str]) -> Dict:
    if not raw_attr:
        return {}
    for candidate in (raw_attr, html.unescape(raw_attr)):
        try:
            loaded = json.loads(candidate)
            if isinstance(loaded, dict):
                return loaded
        except Exception:
            pass
    return {}


class CarsComAdapter(SiteAdapter):
    site_name = "cars_com"
    rate_limit_s = 3.0
    supports_stock_types = ("used", "new", "cpo")

    def build_search_url(self, *, seed_zip: str, radius_miles: int,
                         band_min: Optional[int], band_max: Optional[int],
                         stock_type: str, page: int) -> str:
        query = {
            "zip": seed_zip,
            "maximum_distance": str(radius_miles),
            "sort": "listed_at_desc",
            "page_size": "20",
            "stock_type": stock_type,
        }
        if band_min:
            query["list_price_min"] = str(band_min)
        if band_max is not None:
            query["list_price_max"] = str(band_max)
        if page > 1:
            query["page"] = str(page)
        return f"{BASE_RESULTS_URL}?{urlencode(query)}"

    def parse_cards(self, page_html: str) -> List[Dict]:
        soup = BeautifulSoup(page_html, "html.parser")
        rows: List[Dict] = []
        for card in soup.select(LISTING_SELECTOR):
            payload = _parse_json_attr(card.get("data-vehicle-details"))
            listing_key = card.get("data-listing-id") or payload.get("listingId")
            if not listing_key:
                continue
            link_node = card.select_one("a[data-card-link], a.vehicle-card-link, h2 a")
            href = link_node.get("href") if link_node else None
            row = {
                "site": self.site_name,
                "listing_key": str(listing_key),
                "listing_url": urljoin(SITE_ROOT, href) if href else None,
                "title": get_text(card.select_one("a[data-card-link], h2 a, h2")),
                "year": clean_int(payload.get("year")),
                "make": payload.get("make"),
                "model": payload.get("model"),
                "trim": payload.get("trim"),
                "price": clean_float(payload.get("price")),
                "mileage": clean_int(payload.get("mileage")),
                "body_style": payload.get("bodyStyle"),
                "stock_type": payload.get("stockType"),
                "listing_type": "dealer",
            }
            # Cards show the dealer line ("Best Motors - Renton, WA"); grab what we can.
            card_text = card.get_text(" ", strip=True)
            row.update({k: v for k, v in
                        self.extract_dealer_location(card_text).items()
                        if k != "zip"})  # card rarely shows zip; detail page does
            rows.append(row)
        return rows

    def detail_url(self, row: Dict) -> Optional[str]:
        url = row.get("listing_url")
        if isinstance(url, str) and url.startswith("http"):
            return url
        key = row.get("listing_key")
        return f"{SITE_ROOT}/vehicledetail/{key}/" if key else None

    def parse_detail(self, page_html: str) -> Dict:
        soup = BeautifulSoup(page_html, "html.parser")
        out: Dict = {}

        for dt in soup.find_all("dt"):
            label = (dt.get_text(" ", strip=True) or "").lower().strip().rstrip(":")
            dd = dt.find_next_sibling("dd")
            value = get_text(dd)
            if value and LABEL_MAP.get(label) and not out.get(LABEL_MAP[label]):
                out[LABEL_MAP[label]] = value

        text = soup.get_text(" ", strip=True)
        lower = text.lower()

        vin = self.extract_vin(text)
        if vin:
            out["vin"] = vin
        out.update(self.extract_dealer_location(text))

        if "no accidents" in lower or "no accident" in lower or "0 accidents" in lower:
            out["no_accidents_raw"] = "No accidents reported"
        else:
            m = re.search(r"(\d+)\s+accident", lower)
            if m:
                out["no_accidents_raw"] = f"{m.group(1)} accident(s) reported"

        m = re.search(r"(\d+)\s+owner", lower)
        if m:
            out["owner_count"] = clean_int(m.group(1))
        elif "1-owner" in lower or "one owner" in lower:
            out["owner_count"] = 1

        if "salvage" in lower:
            out["title_status_raw"] = "salvage"
        elif "rebuilt" in lower:
            out["title_status_raw"] = "rebuilt"
        elif "clean title" in lower:
            out["title_status_raw"] = "clean"

        if "certified pre-owned" in lower or "certified preowned" in lower or "cpo" in lower:
            out["cpo_raw"] = "true"

        return out
