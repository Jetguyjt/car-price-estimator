"""
Site-adapter tests against saved HTML fixtures (tests/fixtures/<site>/).

Adapters are pure URL-building + parsing, so these tests pin the canonical
schema contract: if a site redesign breaks a parser, the fixture test won't
catch it (fixtures are frozen) - but if a code change breaks parsing or the
schema, it fails loudly here.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from sites import get_adapter  # noqa: E402
from sites.base import clean_float, clean_int  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def cars_com():
    return get_adapter("cars_com")


def fixture(site, name):
    return (FIXTURES / site / name).read_text()


# --------------------------- URL building ---------------------------

def test_cars_com_search_url_uses_zip_and_radius(cars_com):
    url = cars_com.build_search_url(seed_zip="94110", radius_miles=40,
                                    band_min=20000, band_max=30000,
                                    stock_type="used", page=3)
    assert "zip=94110" in url
    assert "maximum_distance=40" in url
    assert "list_price_min=20000" in url and "list_price_max=30000" in url
    assert "stock_type=used" in url and "page=3" in url


def test_cars_com_search_url_open_band_first_page(cars_com):
    url = cars_com.build_search_url(seed_zip="94110", radius_miles=40,
                                    band_min=80000, band_max=None,
                                    stock_type="used", page=1)
    assert "list_price_min=80000" in url
    assert "list_price_max" not in url and "page=" not in url


# --------------------------- card parsing ---------------------------

def test_cars_com_parse_cards(cars_com):
    rows = cars_com.parse_cards(fixture("cars_com", "results_page.html"))
    assert len(rows) == 2
    g63 = rows[0]
    assert g63["site"] == "cars_com"
    assert g63["listing_key"] == "abc123"
    assert g63["listing_url"] == "https://www.cars.com/vehicledetail/abc123/"
    assert g63["make"] == "Mercedes-Benz" and g63["trim"] == "G 63 4MATIC"
    assert g63["price"] == 132489.0 and g63["mileage"] == 18192
    assert g63["dealer_state"] == "FL"
    assert "zip" not in g63  # cards don't carry a trustworthy zip


def test_cars_com_cards_without_listing_id_skipped(cars_com):
    rows = cars_com.parse_cards(fixture("cars_com", "results_page.html"))
    assert all(r["listing_key"] for r in rows)


# --------------------------- detail parsing ---------------------------

def test_cars_com_parse_detail(cars_com):
    out = cars_com.parse_detail(fixture("cars_com", "detail_page.html"))
    assert out["vin"] == "W1NYC7HJ5MX401234"
    assert out["dealer_city"].endswith("Miami") and out["dealer_state"] == "FL"
    assert out["zip"] == "33137"
    assert out["drivetrain_raw"] == "Four-wheel Drive"
    assert out["engine_raw"].startswith("4.0L V8")
    assert out["no_accidents_raw"] == "No accidents reported"
    assert out["owner_count"] == 1
    assert out["title_status_raw"] == "clean"
    assert out["stock_number"] == "P12345"


def test_detail_url_prefers_listing_url(cars_com):
    assert cars_com.detail_url({"listing_url": "https://x/y"}) == "https://x/y"
    assert cars_com.detail_url({"listing_key": "k1"}) == "https://www.cars.com/vehicledetail/k1/"


# --------------------------- shared helpers ---------------------------

def test_clean_helpers():
    assert clean_int("18,192 mi.") == 18192
    assert clean_float("$132,489") == 132489.0
    assert clean_int(None) is None and clean_float("") is None
