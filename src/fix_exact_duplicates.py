
from pathlib import Path
import hashlib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_DIR / "results" / "dataset_split.csv"
OUTPUT_CSV = PROJECT_DIR / "results" / "dataset_split_clean.csv"

df = pd.read_csv(INPUT_CSV)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# Hash files and identify content present in both splits
hash_to_rows = {}

for idx, row in df.iterrows():
    path = Path(row["image_path"])
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if path.is_file():
        digest = sha256(path)
        hash_to_rows.setdefault(digest, []).append(idx)

moved = 0

for indices in hash_to_rows.values():
    splits = set(df.loc[indices, "split"])

    if "train" in splits and "evaluation" in splits:
        # Keep duplicate content out of evaluation
        for idx in indices:
            if df.at[idx, "split"] == "evaluation":
                df.at[idx, "split"] = "train"
                moved += 1

df.to_csv(OUTPUT_CSV, index=False)

print("Corrected CSV saved to:", OUTPUT_CSV)
print("Evaluation rows moved to training:", moved)
print("\nUpdated split counts:")
print(df["split"].value_counts())
print("\nExact duplicate cross-split groups remaining:",
      sum(
          1 for indices in hash_to_rows.values()
          if len(set(df.loc[indices, "split"])) > 1
      ))
