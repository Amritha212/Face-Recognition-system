
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "baseline_embedding_status.csv"

def main():
    df = pd.read_csv(CSV_PATH)

    low = df[
        df["quality"].astype(str).str.lower().eq("low_quality")
    ]

    print("\nLOW-QUALITY IMAGE SUMMARY")
    print("-" * 40)
    print("Sampled:", len(low))
    print("Successful embeddings:", (low["status"] == "success").sum())
    print("Failed embeddings:", (low["status"] != "success").sum())

    print("\nResults by identity:")
    print(
        low.groupby(["identity", "status"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

if __name__ == "__main__":
    main()
