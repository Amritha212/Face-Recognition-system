
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "results" / "dataset_split_clean.csv"
OUTPUT_CSV = BASE_DIR / "results" / "grouped_split_candidate.csv"

RANDOM_SEED = 42
EVALUATION_RATIO = 0.20

def get_frame_group(path):
    """Extract a frame number from known high-quality filename patterns."""
    name = Path(path).stem

    match = re.search(r"frame[_-]?(\d+)", name, re.IGNORECASE)
    if match:
        frame_number = int(match.group(1))
        # Group nearby frames in blocks of 10.
        return f"frame_block_{frame_number // 10}"

    return None


def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    required = {"image_path", "identity", "quality", "split"}

    if not required.issubset(df.columns):
        raise ValueError(f"Input CSV must contain: {sorted(required)}")

    df["candidate_split"] = df["split"].astype(str).str.lower()
    df["quality_normalized"] = df["quality"].astype(str).str.lower()

    high_mask = df["quality_normalized"].eq("high_quality")
    high = df.loc[high_mask].copy()

    if high.empty:
        raise ValueError("No high-quality images found.")

    high["frame_group"] = high["image_path"].apply(get_frame_group)

    # Rows without a recognizable frame number get their own group.
    missing = high["frame_group"].isna()
    high.loc[missing, "frame_group"] = (
        "single_" + high.loc[missing, "image_path"].astype(str)
    )

    for identity, group in high.groupby("identity"):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=EVALUATION_RATIO,
            random_state=RANDOM_SEED,
        )

        train_indices, eval_indices = next(
            splitter.split(group, groups=group["frame_group"])
        )

        df.loc[group.index[train_indices], "candidate_split"] = "train"
        df.loc[group.index[eval_indices], "candidate_split"] = "evaluation"

    df = df.drop(columns=["quality_normalized"])

    df.to_csv(OUTPUT_CSV, index=False)

    print("Candidate split saved:", OUTPUT_CSV)
    print("Original CSV unchanged:", INPUT_CSV)
    print("\nCandidate split counts:")
    print(df["candidate_split"].value_counts().to_string())

    print("\nCounts by identity, quality, and candidate split:")
    print(
        df.groupby(["identity", "quality", "candidate_split"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print(
        "\nIMPORTANT: This is a candidate split. "
        "Low-quality images retain their previous split assignments. "
        "Cross-quality leakage has not been ruled out."
    )


if __name__ == "__main__":
    main()
