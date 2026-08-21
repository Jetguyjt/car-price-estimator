"""
schema.org JSON-LD extraction for vehicle listings.

Most major car-listing sites (AutoTrader, Edmunds, CarGurus, Cars.com) embed a
schema.org Vehicle/Car/Product object in a <script type="application/ld+json">
block for SEO. Parsing that structured data is far more robust than chasing each
site's ever-changing CSS classes: the JSON-LD schema is stable, standardized, and
the same shape across sites, so one extractor feeds many adapters.

This maps a schema.org Vehicle object onto our canonical listing fields. Adapters
call `listing_from_jsonld(html)` and only need to add site-specific bits (their
listing_key scheme, search URLs) on top.

Ref: https://schema.org/Vehicle , https://schema.org/Car
"""

import json
import re
from typing import Dict, List, Optional

from bs4 import BeautifulSoup

from sites.base import clean_float, clean_int

# schema.org mileage is often {"value": 42000, "unitCode": "SMI"}; drivetrain/fuel
# use controlled vocab we normalize downstream in preprocess_cars_data.
_VEHICLE_TYPES = {"Vehicle", "Car", "MotorizedVehicle", "Motorcycle", "Product"}


def _iter_jsonld(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = tag.string or tag.get_text() or ""
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            continue
        # A block may be a single object, a list, or an @graph container.
        if isinstance(data, dict) and "@graph" in data:
            yield from (d for d in data["@graph"] if isinstance(d, dict))
        elif isinstance(data, list):
            yield from (d for d in data if isinstance(d, dict))
        elif isinstance(data, dict):
            yield data


def _types(obj: Dict) -> set:
    t = obj.get("@type", "")
    return set(t) if isinstance(t, list) else {t}


def find_vehicle(html: str) -> Optional[Dict]:
    """First schema.org object that looks like a vehicle (has make/model or a
    vehicle @type). Returns the raw JSON-LD dict, or None."""
    for obj in _iter_jsonld(html):
        if _types(obj) & _VEHICLE_TYPES:
            if obj.get("brand") or obj.get("model") or obj.get("vehicleIdentificationNumber"):
                return obj
    return None


def _nested(obj, *keys):
    """Follow a chain of keys, tolerating dict-or-scalar (schema.org is loose)."""
    cur = obj
    for k in keys:
        if isinstance(cur, dict):
            cur = cur.get(k)
        else:
            return None
    return cur


def _brand_name(obj) -> Optional[str]:
    b = obj.get("brand") or obj.get("manufacturer")
    if isinstance(b, dict):
        return b.get("name")
    return b if isinstance(b, str) else None


def _price(obj) -> Optional[float]:
    offers = obj.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        return clean_float(offers.get("price") or _nested(offers, "priceSpecification", "price"))
    return None


def _mileage(obj) -> Optional[int]:
    m = obj.get("mileageFromOdometer")
    if isinstance(m, dict):
        return clean_int(m.get("value"))
    return clean_int(m)


def listing_from_jsonld(html: str, *, site: str) -> Optional[Dict]:
    """Canonical listing row from a detail/results page's JSON-LD, or None if absent."""
    v = find_vehicle(html)
    if v is None:
        return None
    color = v.get("color")
    row = {
        "site": site,
        "vin": (v.get("vehicleIdentificationNumber") or "").upper() or None,
        "title": v.get("name"),
        "make": _brand_name(v),
        "model": v.get("model") if isinstance(v.get("model"), str)
                 else _nested(v, "model", "name"),
        "year": clean_int(v.get("vehicleModelDate") or v.get("productionDate")
                          or v.get("modelDate")),
        "trim": v.get("vehicleConfiguration") or v.get("trim"),
        "price": _price(v),
        "mileage": _mileage(v),
        "body_style": v.get("bodyType") if isinstance(v.get("bodyType"), str)
                      else _nested(v, "bodyType", "name"),
        "exterior_color": color if isinstance(color, str) else None,
        "fuel_type_raw": v.get("fuelType") if isinstance(v.get("fuelType"), str)
                         else _nested(v, "fuelType", "name"),
        "drivetrain_raw": v.get("driveWheelConfiguration") if isinstance(
            v.get("driveWheelConfiguration"), str)
            else _nested(v, "driveWheelConfiguration", "name"),
        "transmission_raw": v.get("vehicleTransmission"),
        "engine_raw": _nested(v, "vehicleEngine", "name")
                      or _nested(v, "vehicleEngine", "engineType"),
        "listing_url": v.get("url") or v.get("@id"),
    }
    return {k: val for k, val in row.items() if val is not None}
