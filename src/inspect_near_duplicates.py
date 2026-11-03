
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_split_clean.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "suspected_near_duplicates.csv"

df = pd.read_csv(CSV_PATH)

def average_hash(path, size=8):
    with Image.open(path) as img:
        pixels = np.asarray(
            img.convert("L").resize((size, size))
        )
    return pixels > pixels.mean()

records = []

for split in ["train", "evaluation"]:
    subset = (
        df[df["split"] == split]
        .groupby("identity", group_keys=False)
        .head(20)
    )

    for _, row in subset.iterrows():
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = PROJECT_DIR / path

        if not path.is_file():
            continue

        try:
            records.append({
                "path": str(path),
                "filename": path.name,
                "identity": row["identity"],
                "quality": row["quality"],
                "split": split,
                "hash": average_hash(path),
            })
        except Exception:
            pass

pairs = []

for i, a in enumerate(records):
    for b in records[i + 1:]:
        if a["split"] == b["split"]:
            continue

        distance = int(np.count_nonzero(a["hash"] != b["hash"]))

        if distance <= 5:
            pairs.append({
                "distance": distance,
                "identity_a": a["identity"],
                "quality_a": a["quality"],
                "split_a": a["split"],
                "path_a": a["path"],
                "identity_b": b["identity"],
                "quality_b": b["quality"],
                "split_b": b["split"],
                "path_b": b["path"],
            })

result = pd.DataFrame(pairs)

if not result.empty:
    result = result.sort_values("distance")

result.to_csv(OUTPUT_PATH, index=False)

print("Sample images checked:", len(records))
print("Potential pairs:", len(result))
print("Saved review file:", OUTPUT_PATH)
print("\nPairs by hash distance:")
if result.empty:
    print("No suspicious pairs found in this sample.")
else:
    print(result["distance"].value_counts().sort_index().to_string())
    print("\nPairs involving different identities:")
    print((result["identity_a"] != result["identity_b"]).sum())
    