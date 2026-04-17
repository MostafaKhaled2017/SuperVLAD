from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
from pathlib import Path

from tqdm import tqdm


# This split dictionary is copied from the official MSLS code.
DEFAULT_CITIES = {
    "train": [
        "trondheim",
        "london",
        "boston",
        "melbourne",
        "amsterdam",
        "helsinki",
        "tokyo",
        "toronto",
        "saopaulo",
        "moscow",
        "zurich",
        "paris",
        "bangkok",
        "budapest",
        "austin",
        "berlin",
        "ottawa",
        "phoenix",
        "goa",
        "amman",
        "nairobi",
        "manila",
    ],
    "val": ["cph", "sf"],
    "test": ["miami", "athens", "buenosaires", "stockholm", "bengaluru", "kampala"],
}


WGS84_A = 6378137.0
WGS84_E2 = 0.00669438
WGS84_E_PRIME_SQ = WGS84_E2 / (1 - WGS84_E2)
UTM_K0 = 0.9996


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Format raw Mapillary SLS data into the SuperVLAD evaluation layout."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("datasets/mapillary_sls"),
        help="Root directory of the extracted raw Mapillary SLS dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets/msls"),
        help="Root directory of the formatted MSLS dataset.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="msls",
        help="Dataset name to use under the output root when output-root points to a datasets directory.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy images instead of moving them. This is safer for reruns but uses more disk space.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already exist in the destination.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without copying or moving files.",
    )
    parser.add_argument(
        "--real-test",
        action="store_true",
        help="Require a real metadata-backed test split instead of linking images/test to images/val.",
    )
    return parser.parse_args()


def resolve_output_dataset_root(output_root: Path, dataset_name: str) -> Path:
    if output_root.name == dataset_name:
        return output_root
    return output_root / dataset_name


def latlon_to_utm(lat: float, lon: float) -> tuple[float, float, int, str]:
    zone_number = int((lon + 180) / 6) + 1

    if 56 <= lat < 64 and 3 <= lon < 12:
        zone_number = 32
    if 72 <= lat <= 84:
        if 0 <= lon < 9:
            zone_number = 31
        elif 9 <= lon < 21:
            zone_number = 33
        elif 21 <= lon < 33:
            zone_number = 35
        elif 33 <= lon < 42:
            zone_number = 37

    zone_letter = latitude_to_zone_letter(lat)
    lon_origin = (zone_number - 1) * 6 - 180 + 3

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon_origin_rad = math.radians(lon_origin)

    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat_rad) ** 2)
    t = math.tan(lat_rad) ** 2
    c = WGS84_E_PRIME_SQ * math.cos(lat_rad) ** 2
    a = math.cos(lat_rad) * (lon_rad - lon_origin_rad)

    m = WGS84_A * (
        (1 - WGS84_E2 / 4 - 3 * WGS84_E2**2 / 64 - 5 * WGS84_E2**3 / 256) * lat_rad
        - (3 * WGS84_E2 / 8 + 3 * WGS84_E2**2 / 32 + 45 * WGS84_E2**3 / 1024) * math.sin(2 * lat_rad)
        + (15 * WGS84_E2**2 / 256 + 45 * WGS84_E2**3 / 1024) * math.sin(4 * lat_rad)
        - (35 * WGS84_E2**3 / 3072) * math.sin(6 * lat_rad)
    )

    easting = UTM_K0 * n * (
        a
        + (1 - t + c) * a**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * WGS84_E_PRIME_SQ) * a**5 / 120
    ) + 500000.0

    northing = UTM_K0 * (
        m
        + n
        * math.tan(lat_rad)
        * (
            a**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * a**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * WGS84_E_PRIME_SQ) * a**6 / 720
        )
    )
    if lat < 0:
        northing += 10000000.0

    return easting, northing, zone_number, zone_letter


def latitude_to_zone_letter(latitude: float) -> str:
    if latitude < -80 or latitude > 84:
        raise ValueError(f"Latitude {latitude} is outside the supported UTM range")
    zone_letters = "CDEFGHJKLMNPQRSTUVWX"
    return zone_letters[int((latitude + 80) / 8)]


def build_dst_image_name(
    lat: float,
    lon: float,
    pano_id: str,
    timestamp: str,
    note: str,
) -> str:
    easting, northing, zone_number, zone_letter = latlon_to_utm(lat, lon)
    return (
        f"@{easting:010.2f}"
        f"@{northing:010.2f}"
        f"@{zone_number}"
        f"@{zone_letter}"
        f"@{lat:09.5f}"
        f"@{lon:010.5f}"
        f"@{pano_id}"
        f"@@@@@@{timestamp}"
        f"@{note}"
        f"@.jpg"
    )


def discover_csv_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/*/*/postprocessed.csv"))


def split_for_city(city: str) -> str:
    for split_name, cities in DEFAULT_CITIES.items():
        if city in cities:
            return split_name
    raise ValueError(f"City {city!r} is not present in the known MSLS split mapping")


def iter_rows(postprocessed_csv: Path) -> list[tuple[dict[str, str], dict[str, str]]]:
    raw_csv = postprocessed_csv.with_name("raw.csv")
    with postprocessed_csv.open("r", newline="") as postprocessed_file:
        postprocessed_rows = list(csv.DictReader(postprocessed_file))
    with raw_csv.open("r", newline="") as raw_file:
        raw_rows = list(csv.DictReader(raw_file))
    if len(postprocessed_rows) != len(raw_rows):
        raise ValueError(
            f"Mismatched row counts for {postprocessed_csv} and {raw_csv}: "
            f"{len(postprocessed_rows)} vs {len(raw_rows)}"
        )
    return list(zip(postprocessed_rows, raw_rows))


def parse_is_panorama(value: str) -> bool:
    return value.strip().lower() == "true"


def get_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        if key in row and row[key] != "":
            return row[key]
    raise KeyError(f"None of the keys {keys} were found in row with keys {sorted(row.keys())}")


def get_view_direction(postprocessed_row: dict[str, str]) -> str:
    direction = get_value(postprocessed_row, "view_direction").strip().lower()
    return direction


def get_day_night(postprocessed_row: dict[str, str]) -> str:
    return "night" if get_value(postprocessed_row, "night").strip().lower() == "true" else "day"


def build_destination_dir(dataset_root: Path, split_name: str, csv_dir_name: str) -> Path:
    subset = "database" if csv_dir_name == "database" else "queries"
    return dataset_root / "images" / split_name / subset


def ensure_expected_input(input_root: Path) -> None:
    if not input_root.exists():
        raise FileNotFoundError(f"Input root {input_root} does not exist")


def transfer_file(src: Path, dst: Path, copy_file: bool, dry_run: bool) -> None:
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_file:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def format_msls(
    input_root: Path,
    dataset_root: Path,
    copy_file: bool,
    skip_existing: bool,
    dry_run: bool,
) -> tuple[int, int, set[str]]:
    csv_files = discover_csv_files(input_root)
    if not csv_files:
        raise FileNotFoundError(f"No postprocessed.csv files found under {input_root}")

    moved_count = 0
    skipped_count = 0
    created_splits: set[str] = set()
    for postprocessed_csv in csv_files:
        csv_dir = postprocessed_csv.parent
        city = csv_dir.parent.name
        split_name = split_for_city(city)
        destination_dir = build_destination_dir(dataset_root, split_name, csv_dir.name)
        created_splits.add(split_name)
        row_pairs = iter_rows(postprocessed_csv)

        for postprocessed_row, raw_row in tqdm(row_pairs, desc=f"{city}:{csv_dir.name}"):
            if parse_is_panorama(get_value(raw_row, "is_panorama", "pano")):
                skipped_count += 1
                continue

            pano_id = get_value(raw_row, "pano_id", "key").strip()
            lat = float(get_value(raw_row, "lat"))
            lon = float(get_value(raw_row, "lon"))
            timestamp = get_value(raw_row, "timestamp", "captured_at").replace("-", "").strip()
            note = f"{get_day_night(postprocessed_row)}_{get_view_direction(postprocessed_row)}_{city}"
            image_name = build_dst_image_name(lat, lon, pano_id, timestamp=timestamp, note=note)

            src_image = csv_dir / "images" / f"{pano_id}.jpg"
            if not src_image.exists():
                raise FileNotFoundError(f"Missing source image {src_image}")

            dst_image = destination_dir / image_name
            if dst_image.exists() and skip_existing:
                skipped_count += 1
                continue

            transfer_file(src_image, dst_image, copy_file=copy_file, dry_run=dry_run)
            moved_count += 1

    return moved_count, skipped_count, created_splits


def ensure_test_split(
    dataset_root: Path,
    created_splits: set[str],
    dry_run: bool,
    require_real_test: bool,
) -> None:
    images_root = dataset_root / "images"
    test_dir = images_root / "test"
    val_dir = images_root / "val"

    if "test" in created_splits:
        return
    if require_real_test:
        raise FileNotFoundError(
            "No metadata-backed MSLS test split was found. "
            "Run without --real-test to create images/test as a symlink to images/val."
        )
    if not val_dir.exists() and not dry_run:
        raise FileNotFoundError(f"Validation directory {val_dir} does not exist")
    if test_dir.exists() or test_dir.is_symlink():
        return
    if dry_run:
        return
    test_dir.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(os.path.abspath(val_dir), test_dir)


def main() -> None:
    args = parse_args()
    ensure_expected_input(args.input_root)
    dataset_root = resolve_output_dataset_root(args.output_root, args.dataset_name)
    moved_count, skipped_count, created_splits = format_msls(
        input_root=args.input_root,
        dataset_root=dataset_root,
        copy_file=args.copy,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    ensure_test_split(
        dataset_root=dataset_root,
        created_splits=created_splits,
        dry_run=args.dry_run,
        require_real_test=args.real_test,
    )
    print(f"Formatted dataset at {dataset_root}")
    print(f"Images transferred: {moved_count}")
    print(f"Images skipped: {skipped_count}")


if __name__ == "__main__":
    main()
