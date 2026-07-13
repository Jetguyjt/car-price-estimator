"""
CarPrice — local attribute-enterer front end.

A single-page Streamlit app: pick a car's attributes, get the model's estimated
price, a conformally-calibrated fair-price band, and a deal / fair / overpriced
verdict versus comparable listings. Dropdowns are populated from the exact category
vocabularies the model trained on (`category_levels.json`), so choosing a value the
model never saw surfaces an honest low-confidence warning.

Run:
    streamlit run src/app.py
"""

import json
from pathlib import Path

import streamlit as st

import paths
import predict_listing as pl

# --- vocabularies pulled from the trained model's saved levels ------------------

SENTINELS = {"__unknown__", "__rare__"}
NOT_SPECIFIED = "(not specified)"


def load_levels() -> dict:
    path = paths.ARTIFACT_DIR / "category_levels.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text())


def make_model_maps(levels: dict):
    """Split the make_model vocabulary into a make list and make->models map."""
    makes: dict = {}
    for mm in levels.get("make_model", []):
        if "__" not in mm:
            continue
        make, model = mm.split("__", 1)
        makes.setdefault(make, []).append(model)
    for make in makes:
        makes[make] = sorted(makes[make])
    return makes


def clean_options(levels: dict, key: str) -> list:
    """Real category values for a dropdown, sentinels dropped, sorted."""
    return sorted(v for v in levels.get(key, []) if v not in SENTINELS)


def _titly(s: str) -> str:
    return s.replace("-", " ").title()


def score(listing: dict, asking_price):
    """Thin wrapper so the scoring path is importable/testable without Streamlit."""
    return pl.evaluate(listing, asking_price=asking_price)


# --- UI -------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(page_title="CarPrice — used-car pricing", page_icon="🚗", layout="centered")
    st.title("🚗 CarPrice")
    st.caption(
        "Enter a car's attributes to get a market-relative price band and a "
        "**deal / fair / overpriced** verdict. Estimates are based on **asking prices** "
        "of comparable listings, not realized sale prices."
    )

    levels = load_levels()
    if not levels:
        st.error("No trained model vocabularies found. Run `python3 src/train_price_model.py` first.")
        return

    makes = make_model_maps(levels)
    make_display = {_titly(m): m for m in sorted(makes)}

    # NOTE: no st.form here — the Make→Model dropdowns must cascade live, and inside a
    # form widgets don't rerun until submit (so Model would stay stuck on Acura's list).
    st.subheader("Vehicle")
    c1, c2, c3 = st.columns(3)
    with c1:
        make_disp = st.selectbox("Make", list(make_display.keys()), key="make")
    make = make_display[make_disp]
    with c2:
        model_opts = makes.get(make, [])
        model_display = [_titly(m) for m in model_opts]
        model_disp = st.selectbox("Model", model_display, key="model") if model_display else ""
        model = model_opts[model_display.index(model_disp)] if model_disp else ""
    with c3:
        year = st.number_input("Year", min_value=1990, max_value=2027, value=2020, step=1)

    c4, c5, c6 = st.columns(3)
    with c4:
        mileage = st.number_input("Mileage", min_value=0, max_value=350000, value=40000, step=1000)
    with c5:
        trim = st.selectbox("Trim", [NOT_SPECIFIED] + clean_options(levels, "trim"))
    with c6:
        body_style = st.selectbox("Body style", [NOT_SPECIFIED] + clean_options(levels, "body_style"))

    st.subheader("Drivetrain & condition")
    c7, c8, c9 = st.columns(3)
    with c7:
        drivetrain = st.selectbox("Drivetrain", [NOT_SPECIFIED] + clean_options(levels, "drivetrain"))
    with c8:
        transmission = st.selectbox("Transmission", [NOT_SPECIFIED] + clean_options(levels, "transmission_type"))
    with c9:
        fuel = st.selectbox("Fuel", [NOT_SPECIFIED] + clean_options(levels, "fuel_type_group"))

    c10, c11, c12 = st.columns(3)
    with c10:
        color = st.selectbox("Color", [NOT_SPECIFIED] + clean_options(levels, "car_color"))
    with c11:
        title_status = st.selectbox("Title", [NOT_SPECIFIED] + clean_options(levels, "title_status"))
    with c12:
        accident = st.selectbox("Accident history", [NOT_SPECIFIED] + clean_options(levels, "accident_history"))

    st.subheader("Optional detail (leave blank if unknown)")
    c13, c14, c15 = st.columns(3)
    with c13:
        engine = st.number_input("Engine (L)", min_value=0.0, max_value=8.5, value=None, step=0.1, format="%.1f")
    with c14:
        mpg = st.number_input("MPG", min_value=0, max_value=150, value=None, step=1)
    with c15:
        owner_count = st.number_input("Prior owners", min_value=0, max_value=15, value=None, step=1)

    c16, c17 = st.columns(2)
    with c16:
        cpo = st.checkbox("Certified pre-owned (CPO)")
    with c17:
        is_new = st.checkbox("New (0 prior owners)")

    st.subheader("Price to judge")
    asking_price = st.number_input("Asking price ($)", min_value=0, max_value=1_000_000, value=20000, step=500)

    submitted = st.button("Estimate price", type="primary", use_container_width=True)

    if not submitted:
        return

    def pick(v):
        return None if v == NOT_SPECIFIED else v

    listing = pl.assemble_listing(
        make=make, model=model, year=year, mileage=mileage,
        trim=pick(trim), body_style=pick(body_style), drivetrain=pick(drivetrain),
        transmission_type=pick(transmission), fuel_type_group=pick(fuel),
        car_color=pick(color), title_status=pick(title_status), accident_history=pick(accident),
        engine_size_l=engine, mpg=mpg, owner_count=owner_count, cpo=cpo, is_new=is_new,
    )
    result = score(listing, asking_price if asking_price > 0 else None)
    render_result(result, asking_price, listing)


def render_result(result: dict, asking_price, listing: dict) -> None:
    st.divider()
    point = result["predicted_price"]
    lo, hi = result["fair_range_low"], result["fair_range_high"]

    m1, m2 = st.columns([1, 1])
    with m1:
        delta = None
        if "delta_vs_predicted" in result:
            delta = f"{result['delta_vs_predicted']:+,} vs asking"
        st.metric("Estimated price", f"${point:,.0f}", delta=delta, delta_color="off")
    with m2:
        st.metric("Fair range (q10–q90)", f"${lo:,.0f} – ${hi:,.0f}")

    verdict = result.get("verdict")
    if verdict == "deal":
        st.success(f"✅ **DEAL** — ${asking_price:,.0f} is below the fair range for this car.")
    elif verdict == "overpriced":
        st.error(f"⛔ **OVERPRICED** — ${asking_price:,.0f} is above the fair range for this car.")
    elif verdict == "fair":
        st.info(f"➖ **FAIR** — ${asking_price:,.0f} sits inside the fair range.")

    # Visual band: where the asking price falls within [lo, hi].
    span = max(hi - lo, 1)
    pos = min(max((asking_price - lo) / span, 0.0), 1.0)
    st.caption("Asking price position within the fair range")
    st.progress(pos)

    if result.get("low_confidence"):
        for w in result.get("warnings", []):
            st.warning(f"⚠️ {w}")

    with st.expander("How this was computed"):
        st.write(
            "The model derives engineered features (make_model, car age, miles/year, "
            "mileage bucket, luxury flag) from your inputs, predicts a point price plus a "
            "conformally-calibrated q10–q90 band, and compares your asking price to that band."
        )
        st.json({k: v for k, v in listing.items() if v is not None})


if __name__ == "__main__":
    main()
