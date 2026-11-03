
from pathlib import Path
import pandas as pd
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_split.csv"

df = pd.read_csv(CSV_PATH)

# Test only 2 images per identity from evaluation
sample = (
    df[df["split"] == "evaluation"]
    .groupby("identity", group_keys=False)
    .head(2)
)

print("Testing image readability...")
print("Sample size:", len(sample))

success = 0
failed = 0

for _, row in sample.iterrows():
    image_path = Path(row["image_path"])

    # Handle relative paths from the project directory
    if not image_path.is_absolute():
        image_path = PROJECT_DIR / image_path

    image = cv2.imread(str(image_path))

    if image is not None:
        success += 1
        print("OK:", row["identity"], image_path.name)
    else:
        failed += 1
        print("FAILED:", image_path)

print("\n--- Summary ---")
print("Readable images:", success)
print("Failed images:", failed)
