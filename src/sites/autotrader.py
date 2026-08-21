"""
AutoTrader adapter.

AutoTrader is dealer-heavy like cars.com but with a large, partly-disjoint
inventory (more volume, better geographic spread). Search is zip + radius. Listing
detail pages embed a schema.org Vehicle JSON-LD block, so parsing reuses
sites.schema_org rather than brittle CSS selectors.

NOTE: parse_cards here reads the results page's embedded JSON-LD list; the exact
results-page markup should be validated against a live fetch before a large run
(the URL builder and detail parser are the stable parts). Card-level fields fall
back to the detail pass where absent.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlencode

from sites import schema_org
from sites.base import SiteAdapter

SITE_ROOT = "https://www.autotrader.com"
SEARCH = f"{SITE_ROOT}/cars-for-sale/all-cars"


class AutoTraderAdapter(SiteAdapter):
    site_name = "autotrader"
    rate_limit_s = 4.0
    supports_stock_types = ("used", "new", "cpo")

    def build_search_url(self, *, seed_zip: str, radius_miles: int,
                         band_min: Optional[int], band_max: Optional[int],
                         stock_type: str, page: int) -> str:
        params = {
            "zip": seed_zip,
            "searchRadius": str(radius_miles),
            "sortBy": "datelistedDESC",
            "numRecords": "100",
            "firstRecord": str((page - 1) * 100),
        }
        listing_type = {"used": "USED", "new": "NEW", "cpo": "CERTIFIED"}.get(stock_type)
        if listing_type:
            params["listingType"] = listing_type
        if band_min:
            params["minPrice"] = str(band_min)
        if band_max is not None:
            params["maxPrice"] = str(band_max)
        return f"{SEARCH}?{urlencode(params)}"

    def parse_cards(self, page_html: str) -> List[Dict]:
        # AutoTrader embeds one JSON-LD Vehicle per listing card on results pages.
        rows = []
        for block in schema_org._iter_jsonld(page_html):
            if schema_org._types(block) & schema_org._VEHICLE_TYPES and (
                    block.get("vehicleIdentificationNumber") or block.get("model")):
                row = schema_org.listing_from_jsonld(
                    f'<script type="application/ld+json">{_dumps(block)}</script>',
                    site=self.site_name)
                if row:
                    row["listing_key"] = (row.get("vin")
                                          or _key_from_url(row.get("listing_url")))
                    row.setdefault("listing_type", "dealer")
                    if row["listing_key"]:
                        rows.append(row)
        return rows

    def detail_url(self, row: Dict) -> Optional[str]:
        return row.get("listing_url")

    def parse_detail(self, page_html: str) -> Dict:
        row = schema_org.listing_from_jsonld(page_html, site=self.site_name) or {}
        # keep only detail-enriching fields; identity fields come from the card
        return {k: v for k, v in row.items()
                if k in ("vin", "engine_raw", "drivetrain_raw", "fuel_type_raw",
                         "transmission_raw", "exterior_color", "body_style")}


def _dumps(obj) -> str:
    import json
    return json.dumps(obj)


def _key_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/(\d{6,})(?:[/?]|$)", url)
    return m.group(1) if m else url
