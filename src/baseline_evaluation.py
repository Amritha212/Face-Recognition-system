
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "dataset_split_clean.csv"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cpu")
MAX_IMAGES_PER_IDENTITY = 10
RANDOM_SEED = 42


def load_embedding(image_path, detector, model):
    """Return a normalized FaceNet embedding, or None if detection fails."""
    try:
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        face = detector(image)

        if face is None:
            return None

        with torch.inference_mode():
            embedding = model(face.unsqueeze(0).to(DEVICE))

        vector = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        if not np.isfinite(vector).all():
            return None

        norm = np.linalg.norm(vector)
        if norm == 0:
            return None

        return vector / norm

    except Exception as exc:
        print(f"Could not process {image_path.name}: {exc}")
        return None


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv" 

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    
    

    df = pd.read_csv(CSV_PATH)
    df["candidate_split"] = df["candidate_split"].astype(str).str.lower()
    df["identity"] = df["identity"].astype(str)

    evaluation = df[
        df["candidate_split"].eq("evaluation")
    ].copy()

    if evaluation.empty:
        raise ValueError("No evaluation images found.")

    # Fixed, reproducible small sample for the first test.
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_parts = []

    for identity, group in evaluation.groupby("identity"):
        count = min(MAX_IMAGES_PER_IDENTITY, len(group))
        selected_indices = rng.choice(
            group.index.to_numpy(),
            size=count,
            replace=False,
        )
        sampled_parts.append(group.loc[selected_indices])

    sample = pd.concat(sampled_parts).reset_index(drop=True)

    print("Device:", DEVICE)
    print("Sample images:", len(sample))
    print("Loading pretrained FaceNet...")

    mtcnn = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=torch.device
    )

    model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(DEVICE)

    records = []
    embeddings = []
    labels = []

    for index, row in sample.iterrows():
        path = Path(str(row["image_path"]))
        print(f"[{index + 1}/{len(sample)}] {path.name}")

        vector = load_embedding(path, mtcnn, model)

        status = "success" if vector is not None else "no_face_or_error"

        records.append({
            "image_path": str(path),
            "identity": row["identity"],
            "quality": row.get("quality", "unknown"),
            "status": status,
        })

        if vector is not None:
            embeddings.append(vector)
            labels.append(row["identity"])

    pd.DataFrame(records).to_csv(
        RESULTS_DIR / "baseline_embedding_status.csv",
        index=False,
    )

    if len(embeddings) < 2:
        raise RuntimeError("Too few valid embeddings for pair evaluation.")

    # Each pair gets label 1 for the same identity, otherwise 0.
    pair_scores = []

    for i, j in combinations(range(len(embeddings)), 2):
        similarity = float(np.dot(embeddings[i], embeddings[j]))
        same_identity = int(labels[i] == labels[j])

        pair_scores.append({
            "identity_1": labels[i],
            "identity_2": labels[j],
            "same_identity": same_identity,
            "cosine_similarity": similarity,
        })

    pairs = pd.DataFrame(pair_scores)
    pairs.to_csv(RESULTS_DIR / "baseline_pair_scores.csv", index=False)

    y_true = pairs["same_identity"].to_numpy()
    y_score = pairs["cosine_similarity"].to_numpy()

    print("\nValid embeddings:", len(embeddings))
    print("Pair comparisons:", len(pairs))
    print("Same-person pairs:", int(y_true.sum()))
    print("Different-person pairs:", int((y_true == 0).sum()))

    if len(np.unique(y_true)) < 2:
        raise RuntimeError(
            "Need both same-person and different-person pairs "
            "to calculate ROC and precision-recall metrics."
        )

    # ROC curve and AUC.
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"FaceNet baseline (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline Face Verification ROC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_roc.png", dpi=160)
    plt.close()

    # Precision-recall curve.
    precision, recall, pr_thresholds = precision_recall_curve(
        y_true, y_score
    )
    avg_precision = average_precision_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(
        recall,
        precision,
        label=f"Average precision = {avg_precision:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Baseline Face Verification Precision-Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_precision_recall.png", dpi=160)
    plt.close()

    # Same-person versus different-person score distributions.
    same_scores = pairs.loc[
        pairs["same_identity"] == 1, "cosine_similarity"
    ].to_numpy()
    different_scores = pairs.loc[
        pairs["same_identity"] == 0, "cosine_similarity"
    ].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.hist(
        same_scores, bins=20, alpha=0.6,
        label="Same identity", density=True
    )
    plt.hist(
        different_scores, bins=20, alpha=0.6,
        label="Different identities", density=True
    )
    plt.xlabel("Cosine similarity")
    plt.ylabel("Density")
    plt.title("Baseline Similarity Score Distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "baseline_similarity_distribution.png", dpi=160)
    plt.close()

    # Record a simple operating threshold selected on this sample.
    # This is exploratory and must not be presented as an independent
    # final threshold because the sample is also used for evaluation.
    candidate_thresholds = np.unique(y_score)
    best_threshold = None
    best_accuracy = -1

    for threshold in candidate_thresholds:
        predictions = (y_score >= threshold).astype(int)
        accuracy = float(np.mean(predictions == y_true))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(threshold)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        (y_score >= best_threshold).astype(int),
        labels=[0, 1],
    ).ravel()

    summary = pd.DataFrame([{
        "sample_images": len(sample),
        "valid_embeddings": len(embeddings),
        "failed_embeddings": len(sample) - len(embeddings),
        "pair_count": len(pairs),
        "same_person_pairs": int(y_true.sum()),
        "different_person_pairs": int((y_true == 0).sum()),
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "exploratory_threshold": best_threshold,
        "sample_accuracy_at_exploratory_threshold": best_accuracy,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }])

    summary.to_csv(RESULTS_DIR / "baseline_metrics.csv", index=False)

    print("\nBaseline sample evaluation completed.")
    print(summary.to_string(index=False))
    print("\nSaved results:")
    print("  results/baseline_embedding_status.csv")
    print("  results/baseline_pair_scores.csv")
    print("  results/baseline_metrics.csv")
    print("  results/baseline_roc.png")
    print("  results/baseline_precision_recall.png")
    print("  results/baseline_similarity_distribution.png")
    print(
        "\nIMPORTANT: This is a small exploratory sample, "
        "not the final evaluation."
    )


if __name__ == "__main__":
    main()
