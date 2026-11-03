from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "original"  

if not DATASET_PATH.exists():
    print("Dataset folder not found!")
    print(f"Expected location: {DATASET_PATH.resolve()}")
    raise SystemExit(1)

total_images = 0

print("\nDATASET INSPECTION")
print("=" * 50)

for person_folder in sorted(DATASET_PATH.iterdir()):
    if not person_folder.is_dir():
        continue

    print(f"\nIdentity: {person_folder.name}")

    for quality in ("high_quality", "low_quality"):
        quality_folder = person_folder / quality

        if not quality_folder.exists():
            print(f"  {quality}: Folder missing")
            continue

        images = [
            file for file in quality_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in {
                ".jpg", ".jpeg", ".png", ".bmp", ".webp"
            }
        ]

        total_images += len(images)
        print(f"  {quality}: {len(images)} images")

print("\n" + "=" * 50)
print(f"Total images found: {total_images}")
