
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_split_clean.csv"

df = pd.read_csv(CSV_PATH)

def average_hash(path, size=8):
    with Image.open(path) as img:
        img = img.convert("L").resize((size, size))
        pixels = np.asarray(img)
    return pixels > pixels.mean()

def hash_distance(a, b):
    return int(np.count_nonzero(a != b))

# Small, reproducible sample from each split
sample_parts = []
for split in ["train", "evaluation"]:
    part = df[df["split"] == split].groupby(
        "identity", group_keys=False
    ).head(20)
    sample_parts.append(part)

sample = pd.concat(sample_parts, ignore_index=True)

records = []
for _, row in sample.iterrows():
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if path.is_file():
        try:
            records.append({
                "path": path,
                "identity": row["identity"],
                "split": row["split"],
                "hash": average_hash(path),
            })
        except Exception as exc:
            print("Could not read:", path.name, exc)

print("Readable sample images:", len(records))

pairs = []
for i in range(len(records)):
    for j in range(i + 1, len(records)):
        a, b = records[i], records[j]

        if a["split"] == b["split"]:
            continue

        distance = hash_distance(a["hash"], b["hash"])
        if distance <= 5:
            pairs.append((distance, a, b))

pairs.sort(key=lambda x: x[0])

print("Potential cross-split near-duplicate pairs:", len(pairs))
for distance, a, b in pairs[:20]:
    print(
        f"Distance={distance} | "
        f"{a['identity']} / {a['split']} / {a['path'].name}  <->  "
        f"{b['identity']} / {b['split']} / {b['path'].name}"
    )

print("\nThis is a screening test, not proof of no data leakage.")
