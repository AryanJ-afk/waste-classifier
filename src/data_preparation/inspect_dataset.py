from pathlib import Path

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

DATA_DIR = Path("data/raw")


def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"{DATA_DIR} does not exist")

    class_dirs = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])

    if not class_dirs:
        raise ValueError(f"No class folders found inside {DATA_DIR}")

    print(f"Found {len(class_dirs)} classes:\n")

    total_images = 0

    for class_dir in class_dirs:
        image_files = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS
        ]
        count = len(image_files)
        total_images += count
        print(f"{class_dir.name}: {count} images")

    print(f"\nTotal images: {total_images}")


if __name__ == "__main__":
    main()