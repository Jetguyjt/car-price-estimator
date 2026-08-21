"""Site adapters: one module per listing site, all speaking the canonical schema."""

from typing import Dict

from sites.base import SiteAdapter


def get_adapter(name: str) -> SiteAdapter:
    if name == "cars_com":
        from sites.cars_com import CarsComAdapter
        return CarsComAdapter()
    if name == "ebay_motors":
        from sites.ebay_motors import EbayMotorsAdapter
        return EbayMotorsAdapter()
    if name == "autotrader":
        from sites.autotrader import AutoTraderAdapter
        return AutoTraderAdapter()
    if name == "edmunds":
        from sites.edmunds import EdmundsAdapter
        return EdmundsAdapter()
    raise KeyError(f"unknown site adapter: {name}")


def available_sites() -> list:
    # cars_com and ebay_motors are fully implemented; autotrader/edmunds share the
    # schema.org JSON-LD parser and need a live-HTML validation pass before a big run.
    return ["cars_com", "ebay_motors", "autotrader", "edmunds"]
