"""Merge .jpg/.txt files from nested subdirectories into a flat dataset directory."""

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_SRC = "datasets/video2cls0305/video_2_cls"
DEFAULT_DST = "datasets/full"


def discover_files(src: Path) -> list[Path]:
    """Recursively find all .jpg and .txt files under src."""
    files = []
    for ext in ("*.jpg", "*.txt"):
        files.extend(src.rglob(ext))
    return sorted(files)


def check_collisions(files: list[Path]) -> dict[str, list[Path]]:
    """Return a dict of basename -> list of paths for any duplicate basenames."""
    seen: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        seen[f.name].append(f)
    return {name: paths for name, paths in seen.items() if len(paths) > 1}


def copy_files(files: list[Path], dst: Path, dry_run: bool = False) -> dict[str, int]:
    """Copy all files to dst. Returns count per extension."""
    dst.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = defaultdict(int)
    total = len(files)

    for i, src_file in enumerate(files, 1):
        if not dry_run:
            shutil.copy2(src_file, dst / src_file.name)
        counts[src_file.suffix] += 1
        if i % 2000 == 0 or i == total:
            print(f"  [{i}/{total}] files {'scanned' if dry_run else 'copied'}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge .jpg/.txt files from nested dirs into a flat directory."
    )
    parser.add_argument("--src", default=DEFAULT_SRC, help="Source directory (default: %(default)s)")
    parser.add_argument("--dst", default=DEFAULT_DST, help="Destination directory (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Scan and validate without copying")
    args = parser.parse_args()

    src = Path(args.src).resolve()
    dst = Path(args.dst).resolve()

    if not src.is_dir():
        print(f"Error: source directory not found: {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning source: {src}")
    files = discover_files(src)
    jpg_count = sum(1 for f in files if f.suffix == ".jpg")
    txt_count = sum(1 for f in files if f.suffix == ".txt")
    print(f"Found {jpg_count} .jpg files and {txt_count} .txt files")

    # Collision check
    print("Checking for filename collisions...", end=" ")
    collisions = check_collisions(files)
    if collisions:
        print(f"FAILED ({len(collisions)} duplicates)")
        for name, paths in collisions.items():
            print(f"  {name}:")
            for p in paths:
                print(f"    {p}")
        sys.exit(1)
    print("OK (no duplicates)")

    # Copy
    action = "Scanning" if args.dry_run else "Copying files"
    print(f"{action} to {dst}")
    counts = copy_files(files, dst, dry_run=args.dry_run)

    # Summary
    print("\nDone!")
    for ext, count in sorted(counts.items()):
        print(f"  {ext} files: {count}")
    print(f"  Total: {sum(counts.values())}")


if __name__ == "__main__":
    main()
