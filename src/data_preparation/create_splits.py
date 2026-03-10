from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

RAW_DIR = Path("data/raw")
SPLIT_DIR = Path("data/splits")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42


def get_image_files_and_labels():
    filepaths = []
    labels = []

    class_dirs = sorted([p for p in RAW_DIR.iterdir() if p.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name
        for file_path in class_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
                filepaths.append(file_path)
                labels.append(class_name)

    return filepaths, labels


def clear_split_dir():
    if SPLIT_DIR.exists():
        shutil.rmtree(SPLIT_DIR)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)


def copy_files(files, labels, split_name):
    for file_path, label in zip(files, labels):
        target_dir = SPLIT_DIR / split_name / label
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_dir / file_path.name)


def print_split_counts(labels, split_name):
    print(f"\n{split_name} split:")
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    for class_name in sorted(counts.keys()):
        print(f"{class_name}: {counts[class_name]}")


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"{RAW_DIR} does not exist")

    if abs((TRAIN_RATIO + VAL_RATIO + TEST_RATIO) - 1.0) > 1e-8:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must sum to 1.0")

    filepaths, labels = get_image_files_and_labels()

    if len(filepaths) == 0:
        raise ValueError("No images found in data/raw")

    clear_split_dir()

    test_size_total = VAL_RATIO + TEST_RATIO

    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths,
        labels,
        test_size=test_size_total,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=labels,   # baseline
    )

    val_relative = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - val_relative),
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=y_temp,   # baseline
    )

    copy_files(X_train, y_train, "train")
    copy_files(X_val, y_val, "val")
    copy_files(X_test, y_test, "test")

    print(f"Total images: {len(filepaths)}")
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print_split_counts(y_train, "Train")
    print_split_counts(y_val, "Val")
    print_split_counts(y_test, "Test")

    print("\nDone. Splits saved in data/splits/")


if __name__ == "__main__":
    main()