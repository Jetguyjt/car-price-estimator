# Data expansion: decisions and evidence

Notes on whether to grow the dataset, and how. More data, more sites, and regional
(ZIP3) collection all looked worthwhile, in that priority order. This phase
implemented the storage, quota, and adapter components; the end-to-end runner and
large live collection remained future work.

## 1. Is more data worth collecting? Yes

The learning-curve study (`experiments/run_learning_curve.py`, §7 of
`docs/paper_results.md`) trains on growing subsamples and fits a power law to test MAE.
The curve is still dropping at the current ~10.2k training rows:

| Training rows | Test MAE | Luxury MAE |
|---|---|---|
| 1,000 | ~$5,190 | ~$10,380 |
| 10,191 (all) | ~$2,920 | ~$5,810 |
| 50,000 (projected) | ~$1,990 | - |
| 175,000 (projected) | ~$1,470 | - |

Scaling to ~175k rows projects roughly half the current MAE. The extrapolation is
uncertain (the fitted error floor is optimistic), but the direction is clear, and the
luxury segment (the weakest part of the model) is still improving fastest. This is the
biggest single lever available, and its benefit stacks with the modeling work rather
than competing with it.

## 2. Are other websites worth scraping? Yes

cars.com alone is capped by its pagination limit and is dealer-heavy, so it
under-samples private-seller and enthusiast/luxury listings, which is exactly the
sparse top end where the model is weakest. Adding sites gives:

- Volume and geographic spread (AutoTrader and Edmunds carry large, partly-disjoint
  dealer inventory).
- The private-seller and luxury tail (eBay Motors, later duPont/Hemmings) that a
  dealer-only source can't reach.
- A `source_site` feature: marketplaces price differently (no-haggle vs auction).
- Cross-site VIN dedup as a data-quality check, and a small research angle in how the
  same VIN is priced across sites.

What's implemented: a `SiteAdapter` layer (`src/sites/`) covering

- `cars_com` - the original scraper ported over, now also capturing VIN and dealer
  location;
- `ebay_motors` - the eBay Browse API (structured JSON, VINs, an auction/private flag),
  the easiest source to add; needs a free `EBAY_OAUTH_TOKEN`;
- `autotrader` and `edmunds` - URL builders plus a shared schema.org JSON-LD parser
  (`src/sites/schema_org.py`). Parsing the embedded structured data is sturdier than
  chasing each site's CSS, since the schema is stable.

Parsing is unit-tested against fixtures (`tests/test_multisite.py`). The
AutoTrader/Edmunds results-page card extraction still needs a check against live HTML
before a big run; the URL builders and JSON-LD detail parsers are the stable parts.
CarGurus (heavier bot protection) and the luxury specialists come next.

Skipped: Facebook Marketplace (behind a login, so off the table) and anything else
needing credentialed access. Craigslist is the private-seller stand-in on the list.

## 3. Is regional (ZIP3) collection worth it? Yes, and it unlocks a per-region model

Right now `region` and `zip` are empty on every row (both scrapers used one hardcoded
seed zip at nationwide radius), so region is a dead feature. Geography drives real price
variation (a 4x4 truck in Montana vs Manhattan; climate, local demand), so this is
signal we're throwing away. Collecting by 3-digit ZIP prefix fixes it and is the
prerequisite for the hybrid national + per-region model sketched earlier (one national
model plus per-region corrections, chosen by the buyer's ZIP).

What's implemented (Phase A): the ZIP3 collection system, with its work queue seeded.

- `build_zip3_reference.py` - all 896 ZIP3s with centroids and population;
- `allocate_quotas.py` - population-weighted per-ZIP3 quotas (clipped to 40-1500, target
  175k) plus nationwide sweeps for the sparse luxury bands; 14,339 work units queued;
- SQLite storage and site adapters support per-ZIP3 collection and merging data from separate machines.

Order of work: (1) first, then (2) and (3) together, since the multi-site collection is
already ZIP3-stratified - running it produces regional coverage and cross-site volume in
one pass. The per-region modeling (Phase D) is worth doing once each grouped region has
~6k rows, which the 175k target reaches.

## Operational status

The storage, quota-allocation, and adapter components were implemented and unit-tested. The end-to-end multi-site runner and live HTML checks were not completed in this phase, and the large scrape was not launched.
