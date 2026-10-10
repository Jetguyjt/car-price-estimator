"""Rebuild the three saved figures from the aggregate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


NAVY = "#17324D"
BLUE = "#2E6F9E"
TEAL = "#2A9D8F"
GOLD = "#E9A23B"
RED = "#C75B4D"
LIGHT = "#EAF0F4"
GRAY = "#647482"


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": "#A8B4BF",
            "axes.linewidth": 0.8,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def evaluation_design_schematic(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.1))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.1)
    ax.axis("off")
    ax.text(
        5.5,
        4.75,
        "Same pipeline and data pool; different training coverage",
        ha="center",
        fontsize=17,
        fontweight="bold",
    )
    ax.text(
        5.5,
        4.42,
        "Every listing is scored by a fitted model that did not train on that listing",
        ha="center",
        fontsize=10,
        color=GRAY,
    )

    panels = [
        (0.25, "Random listings", "Categories and ranges\nmostly represented", NAVY),
        (2.95, "Category groups", "Complete test groups\nabsent from training", GOLD),
        (5.65, "Interior range", "Training data remain\non both sides", TEAL),
        (8.35, "Upper tail", "Prediction beyond the\nrepresented range", RED),
    ]
    for left, title, subtitle, color in panels:
        card = FancyBboxPatch(
            (left, 0.55),
            2.4,
            3.35,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor="white",
            edgecolor="#C7D1D9",
            linewidth=1.2,
        )
        ax.add_patch(card)
        ax.text(left + 1.2, 3.55, title, ha="center", fontsize=12, fontweight="bold")
        ax.text(
            left + 1.2,
            3.20,
            subtitle,
            ha="center",
            va="top",
            fontsize=8.1,
            color=GRAY,
            linespacing=1.25,
        )
        y_train, y_test = 2.35, 1.28
        ax.text(
            left + 0.18,
            y_train + 0.22,
            "TRAIN",
            fontsize=7.5,
            color=GRAY,
            fontweight="bold",
        )
        ax.text(
            left + 0.18,
            y_test + 0.22,
            "TEST",
            fontsize=7.5,
            color=GRAY,
            fontweight="bold",
        )
        if title == "Random listings":
            for index in range(12):
                ax.add_patch(
                    Rectangle(
                        (left + 0.70 + 0.125 * index, y_train),
                        0.09,
                        0.32,
                        color=color if index % 4 else LIGHT,
                    )
                )
            for index in range(6):
                ax.add_patch(
                    Rectangle(
                        (left + 0.70 + 0.25 * index, y_test),
                        0.16,
                        0.32,
                        color=color if index % 2 else LIGHT,
                    )
                )
        elif title == "Category groups":
            palette = [BLUE, TEAL, GOLD, RED]
            for index in range(12):
                ax.add_patch(
                    Rectangle(
                        (left + 0.70 + 0.125 * index, y_train),
                        0.09,
                        0.32,
                        color=palette[index % 3],
                    )
                )
            for index in range(6):
                ax.add_patch(
                    Rectangle(
                        (left + 0.70 + 0.25 * index, y_test), 0.16, 0.32, color=RED
                    )
                )
        elif title == "Interior range":
            ax.add_patch(Rectangle((left + 0.70, y_train), 0.48, 0.32, color=TEAL))
            ax.add_patch(Rectangle((left + 1.73, y_train), 0.48, 0.32, color=TEAL))
            ax.add_patch(
                Rectangle(
                    (left + 1.22, y_train),
                    0.47,
                    0.32,
                    facecolor="white",
                    edgecolor=TEAL,
                    hatch="//",
                )
            )
            ax.text(
                left + 1.455,
                y_train + 0.16,
                "ABSENT",
                ha="center",
                va="center",
                fontsize=5.4,
                color=TEAL,
                fontweight="bold",
            )
            ax.add_patch(Rectangle((left + 1.22, y_test), 0.47, 0.32, color=RED))
        else:
            ax.add_patch(Rectangle((left + 0.70, y_train), 1.02, 0.32, color=TEAL))
            ax.add_patch(
                Rectangle(
                    (left + 1.76, y_train),
                    0.45,
                    0.32,
                    facecolor="white",
                    edgecolor=RED,
                    hatch="//",
                )
            )
            ax.text(
                left + 1.985,
                y_train + 0.16,
                "ABSENT",
                ha="center",
                va="center",
                fontsize=5.4,
                color=RED,
                fontweight="bold",
            )
            ax.add_patch(Rectangle((left + 1.76, y_test), 0.45, 0.32, color=RED))
        ax.text(
            left + 1.2,
            0.82,
            "Compare out-of-fold errors",
            ha="center",
            fontsize=8.5,
            color=NAVY,
        )

    for x_position in (2.74, 5.44, 8.14):
        ax.add_patch(
            FancyArrowPatch(
                (x_position, 2.25),
                (x_position + 0.16, 2.25),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#AAB6BF",
            )
        )
    save(fig, output_dir, "evaluation_design_schematic")


def primary_ratio_forest(tables_dir: Path, output_dir: Path) -> None:
    summary = pd.read_csv(tables_dir / "primary_protocols.csv").set_index("protocol")
    order = [
        "holdout_make",
        "holdout_make_model",
        "holdout_age_primary",
        "holdout_mileage_primary",
    ]
    labels = [
        "Manufacturer groups",
        "Make/model categories",
        "Vehicle-age ranges",
        "Mileage ranges",
    ]
    frame = summary.loc[order]
    ratios = frame["mae_ratio"].to_numpy(float)
    low = frame["ratio_ci_low"].to_numpy(float)
    high = frame["ratio_ci_high"].to_numpy(float)
    y_position = np.arange(len(order))
    colors = [RED, GOLD, TEAL, TEAL]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.axvline(1.0, color=NAVY, linewidth=1.2, linestyle="--")
    for index in range(len(order)):
        ax.errorbar(
            ratios[index],
            y_position[index],
            xerr=np.array(
                [[ratios[index] - low[index]], [high[index] - ratios[index]]]
            ),
            fmt="o",
            markersize=8,
            color=colors[index],
            ecolor=colors[index],
            elinewidth=2.2,
            capsize=4,
            zorder=3,
        )
        ax.text(
            high[index] + 0.07,
            y_position[index],
            f"{ratios[index]:.2f}×",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_yticks(y_position, labels)
    ax.set_xlim(0.9, 3.05)
    ax.set_xlabel("MAE relative to random listing-level validation")
    ax.set_title("The apparent error depended on which coverage was withheld")
    ax.grid(axis="x", color=LIGHT)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.45, len(order) - 0.45)
    ax.text(
        1.02,
        len(order) - 0.52,
        "random reference",
        color=GRAY,
        fontsize=8,
        ha="left",
        va="top",
    )
    fig.text(
        0.5,
        -0.02,
        "Points are pooled out-of-fold MAE ratios. Lines are descriptive paired cluster-bootstrap intervals over saved prediction errors, conditional on the fitted models.",
        ha="center",
        color=GRAY,
        fontsize=8.5,
    )
    save(fig, output_dir, "primary_ratio_forest")


def matched_range_comparison(tables_dir: Path, output_dir: Path) -> None:
    frame = pd.read_csv(tables_dir / "matched_ranges.csv")
    label_map = {
        ("mileage", "35,000-59,999"): "Mileage\n35,000–59,999\n(interior)",
        ("mileage", "100,000+"): "Mileage\n100k+\n(tail)",
        ("car_age", "4-5"): "Age\n4–5 years\n(interior)",
        ("car_age", "9+"): "Age\n9+ years\n(tail)",
    }
    frame["display"] = [
        label_map[(row.feature, str(row.band))] for row in frame.itertuples()
    ]
    y_position = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9.3, 4.9))
    for index, row in frame.iterrows():
        ax.plot(
            [row["represented_mae"], row["absent_mae"]],
            [index, index],
            color="#B8C4CD",
            linewidth=3,
            zorder=1,
        )
        ax.scatter(
            row["represented_mae"],
            index,
            s=85,
            color=TEAL,
            zorder=2,
            label="Range represented" if index == 0 else None,
        )
        ax.scatter(
            row["absent_mae"],
            index,
            s=85,
            color=RED,
            zorder=2,
            label="Range absent" if index == 0 else None,
        )
        ax.text(
            row["represented_mae"] - 75,
            index - 0.18,
            f"${row['represented_mae']:,.0f}",
            ha="right",
            va="center",
            fontsize=9,
        )
        ax.text(
            row["absent_mae"] + 75,
            index - 0.18,
            f"${row['absent_mae']:,.0f}",
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
        ax.text(
            (row["represented_mae"] + row["absent_mae"]) / 2,
            index + 0.20,
            f"{row['mae_ratio']:.2f}×",
            ha="center",
            va="center",
            fontsize=9,
            color=NAVY,
            fontweight="bold",
        )
    ax.set_yticks(y_position, frame["display"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean absolute error (dollars)")
    ax.set_title(
        "Matched tests: selected upper tails had larger penalties than selected interiors",
        pad=14,
    )
    ax.grid(axis="x", color=LIGHT)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0, max(frame["absent_mae"]) * 1.22)
    ax.set_ylim(len(frame) - 0.45, -0.45)
    fig.text(
        0.5,
        -0.02,
        "Same test listings and same number of training rows in both conditions; represented condition averaged across three seeded, composition-aware repetitions.",
        ha="center",
        color=GRAY,
        fontsize=9,
    )
    save(fig, output_dir, "matched_range_comparison")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables-dir", type=Path, default=root / "results" / "tables")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    apply_style()
    evaluation_design_schematic(args.output_dir)
    primary_ratio_forest(args.tables_dir, args.output_dir)
    matched_range_comparison(args.tables_dir, args.output_dir)
    print(f"Figures written to {args.output_dir}")


if __name__ == "__main__":
    main()
