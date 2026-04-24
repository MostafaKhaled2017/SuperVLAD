#!/usr/bin/env python3

import argparse
import csv
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class PlaceRecord:
    city: str
    place_id: str
    rows: tuple[dict[str, str], ...]

    @property
    def image_count(self) -> int:
        return len(self.rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reduced, training-compatible sample of the GSV-Cities dataset "
            "using a global image budget while preserving the original directory structure."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("datasets/gsv_cities"),
        help="Path to the source GSV-Cities dataset root.",
    )
    parser.add_argument(
        "--output_dir_name",
        type=str,
        default=None,
        help="Optional name of the output directory created inside dataset_root.",
    )
    parser.add_argument(
        "--total_images",
        type=int,
        required=True,
        help="Maximum total number of images to include in the sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic place ordering.",
    )
    parser.add_argument(
        "--min_images_per_place",
        type=int,
        default=4,
        help="Minimum number of images required for a place to be eligible.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output directory if it already exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.total_images <= 0:
        raise ValueError("--total_images must be positive.")
    if args.min_images_per_place <= 0:
        raise ValueError("--min_images_per_place must be positive.")
    if args.total_images < args.min_images_per_place:
        raise ValueError(
            f"--total_images ({args.total_images}) must be at least "
            f"--min_images_per_place ({args.min_images_per_place}) to sample a valid place."
        )


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir_name = args.output_dir_name or f"sample_{args.total_images}imgs"
    output_root = (dataset_root / output_dir_name).resolve()

    if output_root == dataset_root:
        raise ValueError("Refusing to write the sample into the source dataset root itself.")
    if dataset_root in output_root.parents:
        return dataset_root, output_root
    raise ValueError("The sample output directory must be created inside --dataset_root.")


def validate_dataset_root(dataset_root: Path) -> None:
    dataframes_dir = dataset_root / "Dataframes"
    images_dir = dataset_root / "Images"
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not dataframes_dir.is_dir():
        raise FileNotFoundError(f"Missing Dataframes directory: {dataframes_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing Images directory: {images_dir}")


def load_places(dataset_root: Path, min_images_per_place: int, seed: int) -> Dict[str, List[PlaceRecord]]:
    dataframes_dir = dataset_root / "Dataframes"
    places_by_city: Dict[str, List[PlaceRecord]] = {}

    for csv_path in sorted(dataframes_dir.glob("*.csv")):
        city = csv_path.stem
        with csv_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"CSV file is empty or missing a header: {csv_path}")

            required_columns = {"place_id", "year", "month", "northdeg", "city_id", "lat", "lon", "panoid"}
            missing_columns = required_columns.difference(fieldnames)
            if missing_columns:
                raise ValueError(
                    f"CSV file {csv_path} is missing required columns: {sorted(missing_columns)}"
                )

            grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in reader:
                grouped_rows[row["place_id"]].append(row)

        city_rng = random.Random(f"{seed}:{city}")
        eligible_places: List[PlaceRecord] = []
        for place_id, rows in grouped_rows.items():
            if len(rows) >= min_images_per_place:
                eligible_places.append(PlaceRecord(city=city, place_id=place_id, rows=tuple(rows)))

        # Prefer smaller eligible places to use the image budget efficiently while
        # preserving deterministic randomness among places with the same size.
        eligible_places.sort(key=lambda place: (place.image_count, city_rng.random(), place.place_id))
        places_by_city[city] = eligible_places

    if not places_by_city:
        raise ValueError(f"No CSV files found under {dataframes_dir}")

    return places_by_city


def select_places_balanced(
    places_by_city: Dict[str, List[PlaceRecord]],
    total_images: int,
) -> List[PlaceRecord]:
    selected: List[PlaceRecord] = []
    next_index_by_city = {city: 0 for city in places_by_city}
    remaining_budget = total_images
    cities = sorted(places_by_city)

    while True:
        selected_in_round = False
        any_fit = False

        for city in cities:
            next_index = next_index_by_city[city]
            if next_index >= len(places_by_city[city]):
                continue

            candidate = places_by_city[city][next_index]
            if candidate.image_count <= remaining_budget:
                selected.append(candidate)
                next_index_by_city[city] += 1
                remaining_budget -= candidate.image_count
                selected_in_round = True
                if remaining_budget == 0:
                    return selected
            else:
                any_fit = any_fit or any(
                    place.image_count <= remaining_budget for place in places_by_city[city][next_index:]
                )

        if not selected_in_round:
            if any_fit:
                # Advance past over-budget places so we can still use any smaller remaining places
                # from the same city in later iterations.
                progress = False
                for city in cities:
                    next_index = next_index_by_city[city]
                    while next_index < len(places_by_city[city]) and places_by_city[city][next_index].image_count > remaining_budget:
                        next_index += 1
                    if next_index != next_index_by_city[city]:
                        next_index_by_city[city] = next_index
                        progress = True
                if progress:
                    continue
            break

    return selected


def ensure_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_root}. Pass --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    (output_root / "Dataframes").mkdir(parents=True, exist_ok=False)
    (output_root / "Images").mkdir(parents=True, exist_ok=False)


def get_image_name(row: dict[str, str]) -> str:
    city = row["city_id"]
    place_id = str(row["place_id"]).zfill(7)
    year = str(row["year"]).zfill(4)
    month = str(row["month"]).zfill(2)
    northdeg = str(row["northdeg"]).zfill(3)
    lat = str(row["lat"])
    lon = str(row["lon"])
    panoid = row["panoid"]
    return f"{city}_{place_id}_{year}_{month}_{northdeg}_{lat}_{lon}_{panoid}.jpg"


def write_sample(
    dataset_root: Path,
    output_root: Path,
    places_by_city: Dict[str, List[PlaceRecord]],
    selected_places: Sequence[PlaceRecord],
) -> dict[str, dict[str, int]]:
    selected_by_city: dict[str, list[PlaceRecord]] = defaultdict(list)
    for place in selected_places:
        selected_by_city[place.city].append(place)

    summary: dict[str, dict[str, int]] = {}

    for city, all_places in places_by_city.items():
        source_csv = dataset_root / "Dataframes" / f"{city}.csv"
        target_csv = output_root / "Dataframes" / f"{city}.csv"
        city_places = selected_by_city.get(city, [])

        with source_csv.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            if fieldnames is None:
                raise ValueError(f"CSV file is empty or missing a header: {source_csv}")

        selected_place_ids = {place.place_id for place in city_places}
        rows_to_write: List[dict[str, str]] = []
        copied_images = 0

        if city_places:
            image_dir = output_root / "Images" / city
            image_dir.mkdir(parents=True, exist_ok=True)

            for place in city_places:
                for row in place.rows:
                    rows_to_write.append(row)
                    image_name = get_image_name(row)
                    source_image = dataset_root / "Images" / city / image_name
                    target_image = image_dir / image_name
                    if not source_image.is_file():
                        raise FileNotFoundError(
                            f"Missing image referenced by {source_csv.name}: {source_image}"
                        )
                    shutil.copy2(source_image, target_image)
                    copied_images += 1

        with target_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_to_write)

        summary[city] = {
            "places": len(selected_place_ids),
            "images": copied_images,
            "eligible_places": len(all_places),
        }

    return summary


def print_summary(
    dataset_root: Path,
    output_root: Path,
    total_images_requested: int,
    selected_places: Sequence[PlaceRecord],
    summary_by_city: dict[str, dict[str, int]],
) -> None:
    total_places = len(selected_places)
    total_images = sum(city_summary["images"] for city_summary in summary_by_city.values())

    print(f"Source dataset: {dataset_root}")
    print(f"Sample dataset: {output_root}")
    print(f"Requested image budget: {total_images_requested}")
    print(f"Actual sampled images: {total_images}")
    print(f"Actual sampled places: {total_places}")
    print()
    print("Per-city summary:")
    for city in sorted(summary_by_city):
        city_summary = summary_by_city[city]
        print(
            f"  {city}: places={city_summary['places']} "
            f"images={city_summary['images']} eligible_places={city_summary['eligible_places']}"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_root, output_root = resolve_paths(args)
    validate_dataset_root(dataset_root)

    places_by_city = load_places(
        dataset_root=dataset_root,
        min_images_per_place=args.min_images_per_place,
        seed=args.seed,
    )
    selected_places = select_places_balanced(
        places_by_city=places_by_city,
        total_images=args.total_images,
    )
    if not selected_places:
        raise ValueError(
            "No places could be sampled within the requested image budget. "
            "Increase --total_images or reduce --min_images_per_place."
        )

    ensure_output_root(output_root, overwrite=args.overwrite)
    summary_by_city = write_sample(
        dataset_root=dataset_root,
        output_root=output_root,
        places_by_city=places_by_city,
        selected_places=selected_places,
    )
    print_summary(
        dataset_root=dataset_root,
        output_root=output_root,
        total_images_requested=args.total_images,
        selected_places=selected_places,
        summary_by_city=summary_by_city,
    )


if __name__ == "__main__":
    main()
