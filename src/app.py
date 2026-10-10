"""Local Streamlit interface for the saved vehicle price model.

Run with:
    streamlit run src/app.py
"""

import json
from pathlib import Path

import streamlit as st

import paths
import predict_listing as pl


SENTINELS = {"__unknown__", "__rare__"}
NOT_SPECIFIED = "(not specified)"


def load_levels() -> dict:
    path = paths.ARTIFACT_DIR / "category_levels.json"
    if not path.exists():
        return {}
    return json.loads(Path(path).read_text())


def make_model_maps(levels: dict) -> dict:
    """Split the make_model vocabulary into a make-to-model mapping."""
    makes: dict = {}
    for make_model in levels.get("make_model", []):
        if "__" not in make_model:
            continue
        make, model = make_model.split("__", 1)
        makes.setdefault(make, []).append(model)
    for make in makes:
        makes[make] = sorted(makes[make])
    return makes


def clean_options(levels: dict, key: str) -> list:
    """Return sorted category values without internal sentinel labels."""
    return sorted(value for value in levels.get(key, []) if value not in SENTINELS)


def _display_name(value: str) -> str:
    return value.replace("-", " ").title()


def score(listing: dict, asking_price):
    """Keep the scoring path importable for tests and other interfaces."""
    return pl.evaluate(listing, asking_price=asking_price)


def main() -> None:
    st.set_page_config(page_title="CarPrice", layout="centered")
    st.title("CarPrice")
    st.caption(
        "Enter a vehicle's attributes to get an asking-price estimate and range. "
        "The comparison is based on marketplace asking prices, not completed sales."
    )

    levels = load_levels()
    if not levels:
        st.error("No trained model vocabularies were found in model_artifacts/.")
        return

    makes = make_model_maps(levels)
    reference_year = pl.model_reference_year()
    make_display = {_display_name(make): make for make in sorted(makes)}

    st.subheader("Vehicle")
    c1, c2, c3 = st.columns(3)
    with c1:
        make_label = st.selectbox("Make", list(make_display), key="make")
    make = make_display[make_label]
    with c2:
        model_values = makes.get(make, [])
        model_labels = [_display_name(model) for model in model_values]
        model_label = st.selectbox("Model", model_labels, key="model") if model_labels else ""
        model = model_values[model_labels.index(model_label)] if model_label else ""
    with c3:
        year = st.number_input(
            "Year", min_value=1990, max_value=reference_year + 1, value=2020, step=1
        )

    c4, c5, c6 = st.columns(3)
    with c4:
        mileage = st.number_input("Mileage", min_value=0, max_value=350000, value=40000, step=1000)
    with c5:
        trim = st.selectbox("Trim", [NOT_SPECIFIED] + clean_options(levels, "trim"))
    with c6:
        body_style = st.selectbox(
            "Body style", [NOT_SPECIFIED] + clean_options(levels, "body_style")
        )

    vehicle_status = st.selectbox(
        "Listing status", ["Used", "Certified pre-owned", "New"]
    )
    cpo = vehicle_status == "Certified pre-owned"
    is_new = vehicle_status == "New"

    st.subheader("Drivetrain and condition")
    c7, c8, c9 = st.columns(3)
    with c7:
        drivetrain = st.selectbox(
            "Drivetrain", [NOT_SPECIFIED] + clean_options(levels, "drivetrain")
        )
    with c8:
        transmission = st.selectbox(
            "Transmission",
            [NOT_SPECIFIED] + clean_options(levels, "transmission_type"),
        )
    with c9:
        fuel = st.selectbox(
            "Fuel", [NOT_SPECIFIED] + clean_options(levels, "fuel_type_group")
        )

    c10, c11, c12 = st.columns(3)
    with c10:
        color = st.selectbox(
            "Color", [NOT_SPECIFIED] + clean_options(levels, "car_color")
        )
    with c11:
        title_status = st.selectbox(
            "Title", [NOT_SPECIFIED] + clean_options(levels, "title_status")
        )
    with c12:
        accident = st.selectbox(
            "Accident history",
            [NOT_SPECIFIED] + clean_options(levels, "accident_history"),
        )

    st.subheader("Optional details")
    c13, c14, c15 = st.columns(3)
    with c13:
        engine = st.number_input(
            "Engine (L)", min_value=0.0, max_value=8.5, value=None, step=0.1, format="%.1f"
        )
    with c14:
        mpg = st.number_input("MPG", min_value=0, max_value=150, value=None, step=1)
    with c15:
        owner_count = st.number_input(
            "Prior owners",
            min_value=0,
            max_value=15,
            value=0 if is_new else None,
            step=1,
            disabled=is_new,
        )

    st.subheader("Asking price")
    asking_price = st.number_input(
        "Asking price ($)", min_value=0, max_value=1_000_000, value=20000, step=500
    )

    if not st.button("Estimate price", type="primary", use_container_width=True):
        return

    def optional(value):
        return None if value == NOT_SPECIFIED else value

    listing = pl.assemble_listing(
        make=make,
        model=model,
        year=year,
        mileage=mileage,
        trim=optional(trim),
        body_style=optional(body_style),
        drivetrain=optional(drivetrain),
        transmission_type=optional(transmission),
        fuel_type_group=optional(fuel),
        car_color=optional(color),
        title_status=optional(title_status),
        accident_history=optional(accident),
        engine_size_l=engine,
        mpg=mpg,
        owner_count=owner_count,
        cpo=cpo,
        is_new=is_new,
    )
    result = score(listing, asking_price if asking_price > 0 else None)
    render_result(result, asking_price, listing)


def render_result(result: dict, asking_price: int, listing: dict) -> None:
    st.divider()
    point = result["predicted_price"]
    low = result["fair_range_low"]
    high = result["fair_range_high"]

    c1, c2 = st.columns(2)
    with c1:
        delta = None
        if "asking_price" in result:
            delta = f"${point - result['asking_price']:+,.0f} vs asking"
        st.metric("Estimated price", f"${point:,.0f}", delta=delta, delta_color="off")
    with c2:
        st.metric("Estimated range", f"${low:,.0f} – ${high:,.0f}")

    verdict = result.get("verdict")
    if verdict == "deal":
        st.success(f"**DEAL** — ${asking_price:,.0f} is below the estimated range.")
    elif verdict == "overpriced":
        st.error(f"**OVERPRICED** — ${asking_price:,.0f} is above the estimated range.")
    elif verdict == "fair":
        st.info(f"**FAIR** — ${asking_price:,.0f} is inside the estimated range.")

    if result.get("low_confidence"):
        for warning in result.get("warnings", []):
            st.warning(warning)

    with st.expander("Model inputs"):
        st.json({key: value for key, value in listing.items() if value is not None})


if __name__ == "__main__":
    main()
