
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "results" / "grouped_split_candidate.csv"
FINE_PATH = BASE / "models" / "facenet_finetuned_partial.pt"
OUT = BASE / "results"

SEED = 42
MAX_PER_IDENTITY = 30
DEVICE = torch.device("cpu")


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    evaluation = df[
        df["candidate_split"].astype(str).str.lower().eq("evaluation")
        & df["quality"].astype(str).str.lower().eq("high_quality")
    ].copy()

    # Use a deterministic, identity-balanced subset.
    evaluation = (
        evaluation.sort_values("image_path")
        .groupby("identity", group_keys=False)
        .head(MAX_PER_IDENTITY)
        .reset_index(drop=True)
    )

    print("Evaluation images selected:", len(evaluation))
    print("Loading pretrained and fine-tuned FaceNet models...")

    pretrained = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(DEVICE)

    checkpoint = torch.load(
        FINE_PATH, map_location=DEVICE, weights_only=False
    )
    finetuned = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    finetuned.load_state_dict(checkpoint["model_state_dict"])
    finetuned.eval()

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )

    records = []

    with torch.no_grad():
        for i, row in enumerate(evaluation.itertuples(index=False), 1):
            path = Path(row.image_path)

            try:
                with Image.open(path) as im:
                    face = detector(im.convert("RGB"))

                if face is None:
                    records.append({
                        "image_path": str(path),
                        "identity": row.identity,
                        "status": "no_face_detected",
                    })
                    continue

                face = face.unsqueeze(0).to(DEVICE)

                emb_pre = F.normalize(pretrained(face), p=2, dim=1)[0]
                emb_fine = F.normalize(finetuned(face), p=2, dim=1)[0]

                records.append({
                    "image_path": str(path),
                    "identity": row.identity,
                    "status": "success",
                    "pre_embedding": emb_pre.cpu().numpy(),
                    "fine_embedding": emb_fine.cpu().numpy(),
                })

            except Exception as exc:
                print(f"Error on {path.name}: {exc}")
                records.append({
                    "image_path": str(path),
                    "identity": row.identity,
                    "status": "error",
                })

            if i % 20 == 0:
                print(f"Processed {i}/{len(evaluation)}")

    # Both models must have embeddings for an image to be included.
    valid = [r for r in records if r["status"] == "success"]
    print("Valid embeddings:", len(valid))
    print("Failed images:", len(records) - len(valid))

    if len(valid) < 2 or len({r["identity"] for r in valid}) < 2:
        raise RuntimeError("Not enough valid embeddings for pair evaluation.")

    pair_rows = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            same = int(a["identity"] == b["identity"])

            pre_score = float(np.dot(a["pre_embedding"], b["pre_embedding"]))
            fine_score = float(np.dot(a["fine_embedding"], b["fine_embedding"]))

            pair_rows.append({
                "image_1": a["image_path"],
                "image_2": b["image_path"],
                "identity_1": a["identity"],
                "identity_2": b["identity"],
                "same_identity": same,
                "pretrained_cosine": pre_score,
                "finetuned_cosine": fine_score,
            })

    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(OUT / "pre_post_verification_pair_scores.csv", index=False)

    y = pairs["same_identity"].astype(int)
    if y.nunique() != 2:
        raise RuntimeError("Need both same-identity and different-identity pairs.")

    summary_rows = []

    for model_name, column in [
        ("pretrained", "pretrained_cosine"),
        ("finetuned", "finetuned_cosine"),
    ]:
        scores = pairs[column]
        summary_rows.append({
            "model": model_name,
            "valid_images": len(valid),
            "pair_count": len(pairs),
            "same_identity_pairs": int(y.sum()),
            "different_identity_pairs": int((y == 0).sum()),
            "roc_auc": roc_auc_score(y, scores),
            "average_precision": average_precision_score(y, scores),
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "pre_post_verification_summary.csv", index=False)

    plt.figure()
    for model_name, column in [
        ("Pretrained", "pretrained_cosine"),
        ("Fine-tuned", "finetuned_cosine"),
    ]:
        fpr, tpr, _ = roc_curve(y, pairs[column])
        auc = roc_auc_score(y, pairs[column])
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Pretrained vs Fine-tuned Face Verification ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pre_post_verification_roc.png", dpi=160)
    plt.close()

    plt.figure()
    for model_name, column in [
        ("Pretrained", "pretrained_cosine"),
        ("Fine-tuned", "finetuned_cosine"),
    ]:
        precision, recall, _ = precision_recall_curve(y, pairs[column])
        ap = average_precision_score(y, pairs[column])
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Pretrained vs Fine-tuned Precision-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "pre_post_verification_pr.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model_name, column in [
        (axes[0], "Pretrained", "pretrained_cosine"),
        (axes[1], "Fine-tuned", "finetuned_cosine"),
    ]:
        ax.hist(
            pairs.loc[y == 1, column],
            bins=20, alpha=0.6, label="Same identity",
        )
        ax.hist(
            pairs.loc[y == 0, column],
            bins=20, alpha=0.6, label="Different identities",
        )
        ax.set_title(model_name)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Pair count")
        ax.legend()

    fig.suptitle("Same vs Different Identity Similarity")
    fig.tight_layout()
    fig.savefig(OUT / "pre_post_verification_similarity.png", dpi=160)
    plt.close(fig)

    print("\n--- Verification comparison ---")
    print(summary.to_string(index=False))
    print("\nSaved pair scores, summary, ROC, PR, and similarity plots.")
    print("Evaluation images were not used for gradient training.")


if __name__ == "__main__":
    main()
