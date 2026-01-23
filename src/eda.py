from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MAG_ORDER = ["5-5.9", "6-6.9", "7+"]


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mag_category(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x="mag_category", order=MAG_ORDER, ax=ax)
    ax.set_title("Magnitude Category Distribution")
    ax.set_xlabel("Magnitude Category")
    ax.set_ylabel("Count")
    save_plot(fig, outdir / "eda_mag_category.png")


def plot_depth_distribution(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["depth"], bins=40, kde=True, ax=ax)
    ax.set_title("Depth Distribution")
    ax.set_xlabel("Depth (km)")
    ax.set_ylabel("Count")
    save_plot(fig, outdir / "eda_depth_distribution.png")


def plot_spatial_scatter(df: pd.DataFrame, outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=df,
        x="longitude",
        y="latitude",
        hue="mag_category",
        hue_order=MAG_ORDER,
        s=12,
        alpha=0.5,
        linewidth=0,
        ax=ax,
    )
    ax.set_title("Spatial Distribution by Magnitude")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Magnitude", loc="upper right", frameon=True)
    save_plot(fig, outdir / "eda_spatial_scatter.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA plots.")
    parser.add_argument(
        "--input",
        default="data/processed/earthquake_processed.csv",
        help="Path to processed CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="charts",
        help="Directory to write charts.",
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid")
    df = load_data(args.input)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_mag_category(df, outdir)
    plot_depth_distribution(df, outdir)
    plot_spatial_scatter(df, outdir)

    print(f"Wrote EDA charts to {outdir}")


if __name__ == "__main__":
    main()
