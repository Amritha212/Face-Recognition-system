
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "results" / "baseline_pair_scores.csv"
OUTPUT_DIR = BASE_DIR / "results"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(INPUT_CSV)
    y_true = df["same_identity"].astype(int)
    scores = df["cosine_similarity"].astype(float)

    roc_auc = roc_auc_score(y_true, scores)
    average_precision = average_precision_score(y_true, scores)

    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)

    summary = pd.DataFrame([{
        "pair_count": len(df),
        "same_identity_pairs": int(y_true.sum()),
        "different_identity_pairs": int((y_true == 0).sum()),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }])

    summary.to_csv(
        OUTPUT_DIR / "baseline_verification_summary.csv",
        index=False,
    )

    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline Face Verification ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_verification_roc.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {average_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Baseline Face Verification Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "baseline_verification_precision_recall.png",
        dpi=160,
    )
    plt.close()

    plt.figure()
    same = df.loc[y_true == 1, "cosine_similarity"]
    different = df.loc[y_true == 0, "cosine_similarity"]

    plt.hist(same, bins=20, alpha=0.6, label="Same identity")
    plt.hist(different, bins=20, alpha=0.6, label="Different identities")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Pair count")
    plt.title("Baseline Same vs Different Identity Similarities")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / "baseline_verification_similarity.png",
        dpi=160,
    )
    plt.close()

    print("\n--- Baseline Verification Summary ---")
    print(summary.to_string(index=False))
    print("\nSaved:")
    print(OUTPUT_DIR / "baseline_verification_summary.csv")
    print(OUTPUT_DIR / "baseline_verification_roc.png")
    print(OUTPUT_DIR / "baseline_verification_precision_recall.png")
    print(OUTPUT_DIR / "baseline_verification_similarity.png")
    print("\nNote: preliminary results from the existing 80-image sample.")


if __name__ == "__main__":
    main()
