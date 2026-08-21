"""
eBay Motors adapter via the official Browse API (not HTML scraping).

eBay is the cleanest additional source: a documented REST API returns structured
JSON (VIN, price, mileage, aspects) through its structured JSON data, and its inventory is
private-seller-heavy - exactly the segment cars.com (dealer-heavy) under-covers,
and a good source of enthusiast/luxury cars for the sparse top end.

Auth: the Browse API needs an OAuth application token in EBAY_OAUTH_TOKEN (a free
eBay developer account issues one). Without it, build_search_url still returns the
correct request URL for inspection, but live collection needs the token.

Because responses are JSON, parse_cards/parse_detail here consume the API's JSON
text rather than HTML; the collection runner uses the adapter request semantics.

Refs: https://developer.ebay.com/api-docs/buy/browse/resources/item_summary/methods/search
"""

import json
import os
from typing import Dict, List, Optional
from urllib.parse import urlencode

from sites.base import SiteAdapter, clean_float, clean_int

BROWSE_SEARCH = "https://api.ebay.com/buy/browse/v1/item_summary/search"
BROWSE_ITEM = "https://api.ebay.com/buy/browse/v1/item/"
# eBay Motors > Cars & Trucks
CATEGORY_CARS_TRUCKS = "6001"


class EbayMotorsAdapter(SiteAdapter):
    site_name = "ebay_motors"
    rate_limit_s = 0.2                 # API, not a browser; generous rate limit
    supports_stock_types = ("used",)   # eBay listings are overwhelmingly used/private

    def auth_header(self) -> Dict[str, str]:
        token = os.environ.get("EBAY_OAUTH_TOKEN", "")
        return {"Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                "Content-Type": "application/json"}

    def build_search_url(self, *, seed_zip: str, radius_miles: int,
                         band_min: Optional[int], band_max: Optional[int],
                         stock_type: str, page: int) -> str:
        limit = 200
        params = {
            "category_ids": CATEGORY_CARS_TRUCKS,
            "filter": self._filter(seed_zip, radius_miles, band_min, band_max),
            "limit": str(limit),
            "offset": str((page - 1) * limit),
            "sort": "newlyListed",
        }
        return f"{BROWSE_SEARCH}?{urlencode(params)}"

    @staticmethod
    def _filter(seed_zip, radius_miles, band_min, band_max) -> str:
        parts = [f"deliveryPostalCode:{seed_zip}",
                 f"maxDeliveryCost:0",  # local pickup / in-region
                 "buyingOptions:{FIXED_PRICE|AUCTION}"]
        if band_min or band_max is not None:
            lo = band_min or 0
            hi = band_max if band_max is not None else 10_000_000
            parts.append(f"price:[{lo}..{hi}]")
            parts.append("priceCurrency:USD")
        return ",".join(parts)

    def parse_cards(self, page_json: str) -> List[Dict]:
        """Parse an item_summary/search JSON response into canonical rows."""
        try:
            data = json.loads(page_json)
        except (ValueError, TypeError):
            return []
        rows = []
        for it in data.get("itemSummaries", []) or []:
            aspects = _aspect_map(it)
            price = clean_float(_nested(it, "price", "value"))
            rows.append({
                "site": self.site_name,
                "listing_key": it.get("itemId") or it.get("legacyItemId"),
                "listing_url": it.get("itemWebUrl"),
                "title": it.get("title"),
                "price": price,
                "vin": aspects.get("vehicle identification number (vin)"),
                "year": clean_int(aspects.get("year")),
                "make": aspects.get("make"),
                "model": aspects.get("model"),
                "trim": aspects.get("trim"),
                "mileage": clean_int(aspects.get("mileage")),
                "body_style": aspects.get("body type"),
                "fuel_type_raw": aspects.get("fuel type"),
                "transmission_raw": aspects.get("transmission"),
                "drivetrain_raw": aspects.get("drive type") or aspects.get("drivetrain"),
                "exterior_color": aspects.get("exterior color"),
                "listing_type": "auction" if _is_auction(it) else "private",
                "dealer_state": _nested(it, "itemLocation", "stateOrProvince"),
            })
        return [r for r in rows if r["listing_key"]]

    def detail_url(self, row: Dict) -> Optional[str]:
        key = row.get("listing_key")
        return f"{BROWSE_ITEM}{key}" if key else row.get("listing_url")

    def parse_detail(self, page_json: str) -> Dict:
        """item/{id} JSON -> extra aspects. Card data already covers most fields."""
        try:
            it = json.loads(page_json)
        except (ValueError, TypeError):
            return {}
        aspects = _aspect_map(it, key="localizedAspects")
        out = {}
        if aspects.get("number of cylinders") or aspects.get("engine"):
            out["engine_raw"] = aspects.get("engine") or f"{aspects.get('number of cylinders')} cyl"
        for a, col in [("interior color", "interior_color"),
                       ("number of previous owners", "owner_count")]:
            if aspects.get(a):
                out[col] = aspects[a]
        vin = _find_vin(it)
        if vin:
            out["vin"] = vin
        return out


def _nested(obj, *keys):
    cur = obj
    for k in keys:
        cur = cur.get(k) if isinstance(cur, dict) else None
    return cur


def _aspect_map(item: Dict, key: str = "aspects") -> Dict[str, str]:
    """Flatten eBay's aspect list into a lowercased {name: value} map. Handles both
    item_summary shape (additionalProductIdentities / localizedAspects) shapes."""
    out = {}
    aspects = item.get(key) or item.get("localizedAspects") or []
    if isinstance(aspects, list):
        for a in aspects:
            name = (a.get("name") or a.get("localizedName") or "").strip().lower()
            val = a.get("value") or a.get("localizedValue")
            if name and val:
                out[name] = str(val)
    # item_summary also exposes some fields directly
    return out


def _is_auction(item: Dict) -> bool:
    opts = item.get("buyingOptions") or []
    return "AUCTION" in opts and "FIXED_PRICE" not in opts


def _find_vin(item: Dict) -> Optional[str]:
    for a in item.get("localizedAspects", []) or []:
        if "vin" in (a.get("name", "").lower()):
            return str(a.get("value", "")).upper() or None
    return None
