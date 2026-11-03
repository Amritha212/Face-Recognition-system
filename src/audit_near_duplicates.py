
from pathlib import Path
from collections import defaultdict
import pandas as pd
import cv2
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "results" / "dataset_split_clean.csv"
OUTPUT_PATH = PROJECT_DIR / "results" / "near_duplicate_audit.csv"

# Average-hash distance is a screening signal, not proof of duplication.
MAX_HASH_DISTANCE = 5


def image_hash(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA)
    return resized > resized.mean()


def hash_distance(hash_a, hash_b):
    return int(np.count_nonzero(hash_a != hash_b))


def main():
    df = pd.read_csv(CSV_PATH)
    required = {"image_path", "identity", "split"}

    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")

    # Cache hashes and keep records grouped by identity.
    records = []
    for _, row in df.iterrows():
        path = Path(str(row["image_path"]))
        if not path.is_absolute():
            path = PROJECT_DIR / path

        digest = image_hash(path)
        if digest is not None:
            records.append({
                "identity": str(row["identity"]),
                "split": str(row["split"]).lower(),
                "image_path": str(path),
                "hash": digest,
            })

    by_identity = defaultdict(list)
    for record in records:
        by_identity[record["identity"]].append(record)

    findings = []

    for identity, items in by_identity.items():
        train = [r for r in items if r["split"] == "train"]
        evaluation = [
            r for r in items if "eval" in r["split"] or "test" in r["split"]
        ]

        # Compare each evaluation image with training images of the same identity.
        for eval_item in evaluation:
            for train_item in train:
                distance = hash_distance(
                    eval_item["hash"], train_item["hash"]
                )

                if distance <= MAX_HASH_DISTANCE:
                    findings.append({
                        "identity": identity,
                        "hash_distance": distance,
                        "train_path": train_item["image_path"],
                        "evaluation_path": eval_item["image_path"],
                        "review_status": "needs_visual_review",
                    })

    result = pd.DataFrame(findings)

    if not result.empty:
        result = result.sort_values(
            ["hash_distance", "identity"]
        )

    result.to_csv(OUTPUT_PATH, index=False)

    print("Images successfully hashed:", len(records))
    print("Same-identity cross-split candidate pairs:", len(result))
    print("Audit saved to:", OUTPUT_PATH)

    if not result.empty:
        print("\nCandidates by hash distance:")
        print(result["hash_distance"].value_counts().sort_index())

    print(
        "\nThese are candidates, not confirmed duplicates. "
        "Review them before modifying the split."
    )


if __name__ == "__main__":
    main()
