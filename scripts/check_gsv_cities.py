#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "place_id",
    "city_id",
    "panoid",
    "year",
    "month",
    "northdeg",
    "lat",
    "lon",
}


def _format_int_string(value: str, width: int) -> str:
    return str(int(float(value))).zfill(width)


def build_image_name(row: dict[str, str]) -> str:
    place_id = _format_int_string(row["place_id"], 7)
    year = _format_int_string(row["year"], 4)
    month = _format_int_string(row["month"], 2)
    northdeg = _format_int_string(row["northdeg"], 3)
    city_id = row["city_id"]
    lat = str(row["lat"])
    lon = str(row["lon"])
    panoid = row["panoid"]
    return f"{city_id}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"


def validate_dataset(root: Path, sample_rows_per_csv: int) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    dataframes_dir = root / "Dataframes"
    images_dir = root / "Images"
    if not dataframes_dir.exists():
        raise FileNotFoundError(f"Missing folder: {dataframes_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Missing folder: {images_dir}")

    csv_files = sorted(dataframes_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {dataframes_dir}")

    print(f"Checking GSV-Cities dataset at: {root}")
    print(f"Found {len(csv_files)} CSV metadata files")

    checked_csvs = 0
    checked_rows = 0
    for csv_path in csv_files:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            missing_columns = REQUIRED_COLUMNS - fieldnames
            if missing_columns:
                raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing_columns)}")

            sampled = 0
            for row in reader:
                image_name = build_image_name(row)
                image_path = images_dir / row["city_id"] / image_name
                if not image_path.exists():
                    raise FileNotFoundError(
                        f"Missing image referenced by {csv_path.name}: {image_path}"
                    )
                sampled += 1
                checked_rows += 1
                if sampled >= sample_rows_per_csv:
                    break

            if sampled == 0:
                raise ValueError(f"{csv_path.name} is empty")

        checked_csvs += 1
        print(f"[OK] {csv_path.name}: checked {sampled} row(s)")

    print(
        f"Validation succeeded: {checked_csvs} CSV file(s) checked, "
        f"{checked_rows} sample row(s) resolved to images"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a GSV-Cities dataset directory for SuperVLAD training.")
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the gsv_cities root containing Dataframes/ and Images/",
    )
    parser.add_argument(
        "--sample-rows-per-csv",
        type=int,
        default=1,
        help="How many rows to validate from each city CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_rows_per_csv <= 0:
        raise ValueError("--sample-rows-per-csv must be > 0")
    validate_dataset(args.dataset_root.resolve(), args.sample_rows_per_csv)


if __name__ == "__main__":
    main()
