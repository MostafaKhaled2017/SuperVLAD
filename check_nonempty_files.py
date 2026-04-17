#!/usr/bin/env python3

import argparse
from pathlib import Path

from PIL import Image, UnidentifiedImageError


DEFAULT_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def verify_image_file(path: Path) -> None:
    with Image.open(path) as image:
        image.verify()

    with Image.open(path) as image:
        image.load()


def validate_dataset(
    root: Path,
    verify_images: bool,
    image_extensions: set[str],
) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Missing dataset root: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    zero_byte_files: list[Path] = []
    unreadable_image_files: list[tuple[Path, str]] = []
    checked_files = 0
    checked_images = 0

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        checked_files += 1
        if path.stat().st_size == 0:
            zero_byte_files.append(path)
            continue

        if verify_images and path.suffix.lower() in image_extensions:
            checked_images += 1
            try:
                verify_image_file(path)
            except (OSError, UnidentifiedImageError) as exc:
                unreadable_image_files.append((path, str(exc)))

    print(f"Checked {checked_files} file(s) under: {root}")
    if verify_images:
        print(f"Verified {checked_images} image file(s)")

    if zero_byte_files:
        preview_limit = 50
        print("Found zero-byte files:")
        for path in zero_byte_files[:preview_limit]:
            print(path)
        remaining = len(zero_byte_files) - preview_limit
        if remaining > 0:
            print(f"... and {remaining} more")
        raise ValueError(f"Validation failed: found {len(zero_byte_files)} zero-byte file(s)")

    if unreadable_image_files:
        preview_limit = 50
        print("Found unreadable image files:")
        for path, reason in unreadable_image_files[:preview_limit]:
            print(f"{path}: {reason}")
        remaining = len(unreadable_image_files) - preview_limit
        if remaining > 0:
            print(f"... and {remaining} more")
        raise ValueError(
            "Validation failed: found "
            f"{len(unreadable_image_files)} unreadable image file(s)"
        )

    print("Validation succeeded: no zero-byte files found")
    if verify_images:
        print("Image verification succeeded: all matching image files could be decoded")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively verify that a dataset directory does not contain zero-byte files "
            "and optionally check that image files can be decoded."
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Path to the dataset directory to scan recursively.",
    )
    parser.add_argument(
        "--verify-images",
        action="store_true",
        help="Attempt to decode image files with Pillow.",
    )
    parser.add_argument(
        "--image-extensions",
        nargs="+",
        default=sorted(DEFAULT_IMAGE_EXTENSIONS),
        help=(
            "Image file extensions to verify when --verify-images is enabled. "
            "Example: --image-extensions .jpg .jpeg .png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_extensions = {
        extension if extension.startswith(".") else f".{extension}"
        for extension in args.image_extensions
    }
    validate_dataset(
        args.dataset_root.resolve(),
        verify_images=args.verify_images,
        image_extensions={extension.lower() for extension in image_extensions},
    )


if __name__ == "__main__":
    main()
