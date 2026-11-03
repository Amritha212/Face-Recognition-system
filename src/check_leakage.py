
from pathlib import Path
import hashlib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_split.csv"

df = pd.read_csv(CSV_PATH)

def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

print("Checking exact duplicate files...")

hashes = {}
for _, row in df.iterrows():
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.is_file():
        continue

    digest = file_hash(path)
    hashes.setdefault(digest, []).append(
        (row["identity"], row["split"], str(path))
    )

cross_split_duplicates = []

for digest, entries in hashes.items():
    splits = {entry[1] for entry in entries}
    if len(splits) > 1:
        cross_split_duplicates.append(entries)

print("Unique file hashes:", len(hashes))
print("Exact duplicate groups across splits:", len(cross_split_duplicates))

for group in cross_split_duplicates[:10]:
    print("\nDuplicate group:")
    for identity, split, path in group:
        print(f"  {split:10} {identity:10} {Path(path).name}")

if not cross_split_duplicates:
    print("No exact duplicate files found across splits.")
    print("Note: near-duplicate video frames still need separate checking.")
