"""
Edmunds adapter.

Edmunds carries dealer inventory plus editorial/trim metadata, and - like the
others - embeds schema.org Vehicle JSON-LD on detail pages, so parsing reuses
sites.schema_org. Search is zip + radius.

Same caveat as the AutoTrader adapter: the URL builder and JSON-LD detail parser
are the stable parts; the results-page card extraction should be validated against
a live fetch before a large collection run.
"""

import re
from typing import Dict, List, Optional
from urllib.parse import urlencode

from sites import schema_org
from sites.base import SiteAdapter

SITE_ROOT = "https://www.edmunds.com"
SEARCH = f"{SITE_ROOT}/inventory/srp.html"


class EdmundsAdapter(SiteAdapter):
    site_name = "edmunds"
    rate_limit_s = 4.0
    supports_stock_types = ("used", "cpo")

    def build_search_url(self, *, seed_zip: str, radius_miles: int,
                         band_min: Optional[int], band_max: Optional[int],
                         stock_type: str, page: int) -> str:
        params = {
            "zip": seed_zip,
            "radius": str(radius_miles),
            "sort": "publishDate:desc",
            "pageNum": str(page),
            "inventoryType": "used,cpo" if stock_type == "cpo" else "used",
        }
        if band_min:
            params["priceMin"] = str(band_min)
        if band_max is not None:
            params["priceMax"] = str(band_max)
        return f"{SEARCH}?{urlencode(params)}"

    def parse_cards(self, page_html: str) -> List[Dict]:
        rows = []
        for block in schema_org._iter_jsonld(page_html):
            if schema_org._types(block) & schema_org._VEHICLE_TYPES and (
                    block.get("vehicleIdentificationNumber") or block.get("model")):
                import json
                row = schema_org.listing_from_jsonld(
                    f'<script type="application/ld+json">{json.dumps(block)}</script>',
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
        return {k: v for k, v in row.items()
                if k in ("vin", "engine_raw", "drivetrain_raw", "fuel_type_raw",
                         "transmission_raw", "exterior_color", "body_style")}


def _key_from_url(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    m = re.search(r"/(\d{6,}|[a-f0-9-]{20,})(?:[/?]|$)", url)
    return m.group(1) if m else url
