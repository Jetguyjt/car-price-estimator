"""Site adapters: one module per listing site, all speaking the canonical schema."""

from typing import Dict

from sites.base import SiteAdapter


def get_adapter(name: str) -> SiteAdapter:
    if name == "cars_com":
        from sites.cars_com import CarsComAdapter
        return CarsComAdapter()
    raise KeyError(f"unknown site adapter: {name}")


def available_sites() -> list:
    return ["cars_com"]
