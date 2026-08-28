"""
Build data/reference/zip3_reference.csv - the ZIP3 table that drives regional
collection quotas and, later, dynamic region grouping.

One row per 3-digit ZIP prefix (~900 in the US):
    zip3, seed_zip5, lat, lon, population, n_zctas

- Centroids come from the Census ZCTA Gazetteer (free, ~1MB zip).
- Population comes from the ACS 5-year API (B01003_001E, all ZCTAs in one call).
- zip3 lat/lon is the population-weighted mean of its member ZCTAs.
- seed_zip5 is the most populous member ZCTA - the zip we hand to a site's
  "search near <zip>" parameter to anchor that ZIP3's searches.

This is a run-once script; the output CSV is committed so nothing downstream
depends on census.gov being up.

Run:
    python3 src/build_zip3_reference.py
"""

import io
import json
import os
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import pandas as pd

import paths

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120, context=_SSL_CTX) as resp:
        return resp.read()

GAZETTEER_URL = (
    "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/"
    "2023_Gazetteer/2023_Gaz_zcta_national.zip"
)
# Population per ZIP. The Census API requires a (free) key nowadays; when
# CENSUS_API_KEY is set we use real ACS population. Otherwise we fall back to
# the keyless IRS SOI ZIP file and use N2 (number of individuals on returns) -
# an excellent *relative* population proxy, which is all quotas/centroids need.
ACS_POP_URL = (
    "https://api.census.gov/data/2022/acs/acs5"
    "?get=B01003_001E&for=zip%20code%20tabulation%20area:*&key={key}"
)
IRS_SOI_URL = "https://www.irs.gov/pub/irs-soi/21zpallnoagi.csv"

REFERENCE_DIR = Path(paths.DATA_DIR) / "reference"
OUTPUT_CSV = REFERENCE_DIR / "zip3_reference.csv"


def fetch_gazetteer() -> pd.DataFrame:
    print(f"downloading {GAZETTEER_URL} ...")
    blob = _get(GAZETTEER_URL)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        name = zf.namelist()[0]
        df = pd.read_csv(zf.open(name), sep="\t", dtype={"GEOID": str})
    df.columns = [c.strip() for c in df.columns]
    return df[["GEOID", "INTPTLAT", "INTPTLONG"]].rename(
        columns={"GEOID": "zcta", "INTPTLAT": "lat", "INTPTLONG": "lon"}
    )


def fetch_population() -> pd.DataFrame:
    key = os.environ.get("CENSUS_API_KEY")
    if key:
        url = ACS_POP_URL.format(key=key)
        print("downloading ACS population (with key) ...")
        data = json.loads(_get(url))
        df = pd.DataFrame(data[1:], columns=data[0])
        df = df.rename(columns={"B01003_001E": "population",
                                "zip code tabulation area": "zcta"})
    else:
        print(f"downloading {IRS_SOI_URL} (keyless population proxy) ...")
        df = pd.read_csv(io.BytesIO(_get(IRS_SOI_URL)), dtype={"ZIPCODE": str},
                         usecols=["ZIPCODE", "N2"])
        df = df[~df["ZIPCODE"].isin(["00000", "99999"])]
        df = df.rename(columns={"ZIPCODE": "zcta", "N2": "population"})
        df["zcta"] = df["zcta"].str.zfill(5)
    df["population"] = pd.to_numeric(df["population"], errors="coerce").fillna(0)
    return df.groupby("zcta", as_index=False)["population"].sum()


def build() -> pd.DataFrame:
    gaz = fetch_gazetteer()
    pop = fetch_population()
    z = gaz.merge(pop, on="zcta", how="left")
    z["population"] = z["population"].fillna(0)
    z["zip3"] = z["zcta"].str[:3]

    def agg(group: pd.DataFrame) -> pd.Series:
        w = group["population"].clip(lower=1)  # avoid zero-weight degenerate centroids
        seed = group.loc[group["population"].idxmax(), "zcta"]
        return pd.Series({
            "seed_zip5": seed,
            "lat": (group["lat"] * w).sum() / w.sum(),
            "lon": (group["lon"] * w).sum() / w.sum(),
            "population": int(group["population"].sum()),
            "n_zctas": len(group),
        })

    out = z.groupby("zip3").apply(agg, include_groups=False).reset_index()
    out = out.sort_values("zip3").reset_index(drop=True)
    return out


def main() -> None:
    out = build()
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"wrote {len(out)} zip3 rows -> {OUTPUT_CSV}")
    print(f"total population covered: {out['population'].sum():,}")
    print(out.nlargest(5, "population").to_string(index=False))


if __name__ == "__main__":
    main()
