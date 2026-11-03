
from pathlib import Path
from PIL import Image
import csv
import re

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset" / "original"
RESULTS_DIR = PROJECT_DIR / "results"
OUTPUT_CSV = RESULTS_DIR / "dataset_split.csv"

# --------------------------------------------------
# 2. CONFIGURATION
# --------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}

TRAIN_RATIO = 0.80

# --------------------------------------------------
# 3. NUMERIC FILENAME SORTING
# --------------------------------------------------

def numeric_sort_key(path):
    """
    Sort filenames such as:
    t (1).png, t (2).png, t (10).png

    Files without numbers are sorted by filename.
    """
    match = re.search(r"\((\d+)\)", path.stem)

    if match:
        return (0, int(match.group(1)), path.name.lower())

    return (1, 0, path.name.lower())


# --------------------------------------------------
# 4. CHECK WHETHER AN IMAGE IS READABLE
# --------------------------------------------------

def is_valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


# --------------------------------------------------
# 5. MAIN DATASET SPLITTING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 55)
    print("FACE RECOGNITION DATASET PREPARATION")
    print("=" * 55)

    if not DATASET_DIR.is_dir():
        print("\nERROR: Dataset folder not found.")
        print(f"Expected location: {DATASET_DIR}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    unreadable_images = []
    identity_counts = {}

    # Each immediate subfolder represents one identity.
    identity_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not identity_folders:
        print("ERROR: No identity folders found.")
        return

    # --------------------------------------------------
    # 6. PROCESS EACH IDENTITY
    # --------------------------------------------------

    for identity_folder in identity_folders:

        identity = identity_folder.name

        print(f"\nProcessing identity: {identity}")

        identity_counts[identity] = {
            "train": 0,
            "evaluation": 0
        }

        for quality in ("high_quality", "low_quality"):

            quality_folder = identity_folder / quality

            if not quality_folder.is_dir():
                print(f"  {quality}: folder not found; skipping")
                continue

            image_paths = sorted(
                (
                    path for path in quality_folder.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in IMAGE_EXTENSIONS
                ),
                key=numeric_sort_key
            )

            # Validate images before assigning splits.
            valid_images = []

            for path in image_paths:
                if is_valid_image(path):
                    valid_images.append(path)
                else:
                    unreadable_images.append(str(path))

            total = len(valid_images)

            if total == 0:
                print(f"  {quality}: no readable images")
                continue

            # Avoid an empty split for folders with at least
            # two readable images.
            if total == 1:
                print(
                    f"  WARNING: {identity}/{quality} has only "
                    "one readable image; assigning it to training."
                )
                train_count = 1
            else:
                train_count = int(total * TRAIN_RATIO)
                train_count = max(1, min(train_count, total - 1))

            print(
                f"  {quality}: {total} readable images; "
                f"{train_count} training, "
                f"{total - train_count} evaluation"
            )

            # --------------------------------------------------
            # 7. ASSIGN SPLITS
            # --------------------------------------------------

            for index, image_path in enumerate(valid_images):

                if index < train_count:
                    split = "train"
                else:
                    split = "evaluation"

                rows.append({
                    "image_path": str(image_path.resolve()),
                    "identity": identity,
                    "quality": quality,
                    "split": split
                })

                identity_counts[identity][split] += 1

    # --------------------------------------------------
    # 8. SAVE CSV MANIFEST
    # --------------------------------------------------

    fieldnames = [
        "image_path",
        "identity",
        "quality",
        "split"
    ]

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # 9. PRINT SUMMARY
    # --------------------------------------------------

    training_total = sum(
        1 for row in rows if row["split"] == "train"
    )

    evaluation_total = sum(
        1 for row in rows if row["split"] == "evaluation"
    )

    print("\n" + "=" * 55)
    print("DATASET SPLIT SUMMARY")
    print("=" * 55)

    for identity, counts in identity_counts.items():
        print(
            f"{identity}: "
            f"training={counts['train']}, "
            f"evaluation={counts['evaluation']}"
        )

    print("-" * 55)
    print(f"Total usable images: {len(rows)}")
    print(f"Training images: {training_total}")
    print(f"Evaluation images: {evaluation_total}")
    print(f"Unreadable images: {len(unreadable_images)}")
    print(f"\nCSV manifest saved to:\n{OUTPUT_CSV}")

    if unreadable_images:
        print("\nFirst 10 unreadable images:")
        for path in unreadable_images[:10]:
            print(path)

    print("\nDataset preparation completed.")


# --------------------------------------------------
# 10. RUN SCRIPT
# --------------------------------------------------

if __name__ == "__main__":
    main()
