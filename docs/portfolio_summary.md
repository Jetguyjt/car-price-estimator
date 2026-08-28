# Portfolio summary

_Point-in-time planning note from the initial repository setup._

## One-sentence summary

CarPrice is an end-to-end automotive ML research project that collects real used-car listing data, builds a traceable dataset, and trains interpretable price models that estimate whether a listing is cheap, fair, or expensive relative to comparable market listings.

## What this project demonstrates

- **Data engineering:** raw listing collection, detail-page enrichment, schema consistency, deduplication, and provenance tracking.
- **Machine learning:** LightGBM regression, quantile models for prediction intervals, categorical feature handling, and held-out evaluation.
- **Evaluation discipline:** comparison against baseline methods, segment error analysis, calibration checks, and model limitation reporting.
- **Explainability:** feature importance and SHAP summaries for understanding which variables drive predictions.
- **Product thinking:** single-listing scoring flow that returns a market-relative price band and confidence warnings for unseen categories.

## Why it is realistic

The project intentionally avoids claiming more than the data supports:

- The target is listing **asking price**, not final sale price.
- Missing listing-detail fields are left blank rather than inferred.
- The model reports uncertainty and known weak segments.
- The roadmap prioritizes data quality and external benchmarking before any production-style valuation claims.

## Current limitations

- Dataset size is still modest for a final production model.
- Some vehicle-history fields are sparsely visible on listing pages.
- Luxury and rare vehicles have higher error because comparable examples are less common.
- Asking prices can include dealer strategy and regional effects that are not yet fully modeled.

## Next improvements

1. Scale raw listings to 10k-25k unique rows while preserving provenance.
2. Improve detail-page extraction and field-level coverage reporting.
3. Add automated dataset validation before training.
4. Re-train models on the expanded dataset and compare performance against the current baseline.
5. Build a small web interface around `predict_listing.py` for portfolio presentation.
