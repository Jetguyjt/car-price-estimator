"""
Tests for the schema.org JSON-LD extractor and the multi-site adapters.

The JSON-LD path is what makes AutoTrader/Edmunds/CarGurus tractable from one
parser, and the eBay Browse API adapter is a JSON (not HTML) source; both are
pinned here against representative fixtures.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sites import get_adapter  # noqa: E402
from sites import schema_org  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _fx(*p):
    return (FIXTURES.joinpath(*p)).read_text()


# --------------------------- schema.org JSON-LD ---------------------------

def test_jsonld_vehicle_extraction():
    row = schema_org.listing_from_jsonld(_fx("schema_org", "vehicle_detail.html"),
                                         site="autotrader")
    assert row["make"] == "Mercedes-Benz"
    assert row["model"] == "AMG G 63"
    assert row["year"] == 2021
    assert row["vin"] == "W1NYC7HJ5MX401234"      # uppercased
    assert row["price"] == 132489.0
    assert row["mileage"] == 18192
    assert row["body_style"] == "SUV"
    assert row["engine_raw"] == "4.0L V8 Twin Turbo"
    assert row["fuel_type_raw"] == "Gasoline"


def test_jsonld_absent_returns_none():
    assert schema_org.listing_from_jsonld("<html><body>no jsonld</body></html>",
                                          site="edmunds") is None


# --------------------------- AutoTrader / Edmunds ---------------------------

def test_autotrader_url_and_card_parse():
    a = get_adapter("autotrader")
    url = a.build_search_url(seed_zip="94110", radius_miles=40, band_min=20000,
                             band_max=30000, stock_type="used", page=2)
    assert "zip=94110" in url and "searchRadius=40" in url
    assert "minPrice=20000" in url and "maxPrice=30000" in url and "firstRecord=100" in url
    rows = a.parse_cards(_fx("schema_org", "vehicle_detail.html"))
    assert len(rows) == 1
    assert rows[0]["make"] == "Mercedes-Benz"
    assert rows[0]["listing_key"] == "W1NYC7HJ5MX401234"   # VIN preferred as key


def test_edmunds_url_build():
    e = get_adapter("edmunds")
    url = e.build_search_url(seed_zip="10001", radius_miles=50, band_min=None,
                             band_max=None, stock_type="cpo", page=3)
    assert "zip=10001" in url and "radius=50" in url and "pageNum=3" in url
    assert "used%2Ccpo" in url    # urlencoded "used,cpo"


# --------------------------- eBay Browse API ---------------------------

def test_ebay_search_url():
    e = get_adapter("ebay_motors")
    url = e.build_search_url(seed_zip="94110", radius_miles=40, band_min=80000,
                             band_max=120000, stock_type="used", page=1)
    assert "api.ebay.com/buy/browse" in url
    assert "deliveryPostalCode%3A94110" in url or "deliveryPostalCode:94110" in url
    assert "80000..120000" in url or "80000..120000".replace("..", "..") in url


def test_ebay_parse_cards():
    e = get_adapter("ebay_motors")
    rows = e.parse_cards(_fx("ebay_motors", "search.json"))
    assert len(rows) == 2
    car = rows[0]
    assert car["make"] == "Porsche" and car["model"] == "911"
    assert car["year"] == 2019 and car["mileage"] == 24000
    assert car["vin"] == "WP0AB2A99KS123456"
    assert car["price"] == 98500.0
    assert car["listing_type"] == "private"
    assert rows[1]["listing_type"] == "auction"     # AUCTION-only -> auction
    assert car["dealer_state"] == "CA"


def test_ebay_detail_url():
    e = get_adapter("ebay_motors")
    assert e.detail_url({"listing_key": "v1|3456789|0"}).endswith("v1|3456789|0")
