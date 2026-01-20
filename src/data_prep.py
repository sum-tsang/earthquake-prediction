from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse time and numeric fields
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for col in ["latitude", "longitude", "depth", "mag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["time", "latitude", "longitude", "depth", "mag"])

    # Time-based features
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["day"] = df["time"].dt.day
    df["dayofyear"] = df["time"].dt.dayofyear

    # Magnitude category target
    bins = [5.0, 6.0, 7.0, 10.1]
    labels = ["5-5.9", "6-6.9", "7+"]
    df["mag_category"] = pd.cut(df["mag"], bins=bins, labels=labels, right=False)

    feature_cols = [
        "latitude",
        "longitude",
        "depth",
        "year",
        "month",
        "day",
        "dayofyear",
    ]
    return df[feature_cols + ["mag_category"]].dropna()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare earthquake dataset.")
    parser.add_argument(
        "--input",
        default="data/raw/earthquake1826_2026.csv",
        help="Path to raw CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/earthquake_processed.csv",
        help="Path to write processed CSV file.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    processed = build_features(df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    print(f"Wrote {len(processed)} rows to {output_path}")


if __name__ == "__main__":
    main()
