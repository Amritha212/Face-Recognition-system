
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv"
CHECKPOINT_PATH = BASE_DIR / "models" / "identity_classifier.pt"
OUTPUT_DIR = BASE_DIR / "results"
OUTPUT_CSV = OUTPUT_DIR / "identity_classifier_predictions.csv"

DEVICE = torch.device("cpu")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    evaluation = df[
        df["candidate_split"].astype(str).str.lower().eq("evaluation")
        & df["quality"].astype(str).str.lower().eq("high_quality")
    ].copy()

    # Limit evaluation to at most 30 images per identity for this first test.
    evaluation = (
        evaluation.groupby("identity", group_keys=False)
        .head(30)
        .reset_index(drop=True)
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    label_map = checkpoint["label_map"]
    reverse_map = {value: key for key, value in label_map.items()}

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )

    backbone = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    classifier = nn.Linear(
        checkpoint["embedding_dimension"], len(label_map)
    ).to(DEVICE)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()

    rows = []
    y_true = []
    y_pred = []

    with torch.no_grad():
        for i, row in enumerate(evaluation.itertuples(index=False), start=1):
            path = Path(row.image_path)
            actual = str(row.identity)

            try:
                with Image.open(path) as source:
                    face = detector(source.convert("RGB"))

                if face is None:
                    rows.append({
                        "image_path": str(path),
                        "identity": actual,
                        "predicted_identity": "",
                        "status": "no_face_detected",
                        "correct": False,
                    })
                    continue

                embedding = backbone(face.unsqueeze(0).to(DEVICE))
                embedding = nn.functional.normalize(embedding, p=2, dim=1)
                logits = classifier(embedding)
                predicted = reverse_map[int(logits.argmax(dim=1).item())]

                correct = predicted == actual
                rows.append({
                    "image_path": str(path),
                    "identity": actual,
                    "predicted_identity": predicted,
                    "status": "success",
                    "correct": correct,
                })
                y_true.append(actual)
                y_pred.append(predicted)

            except Exception as exc:
                print(f"Error processing {path.name}: {exc}")
                rows.append({
                    "image_path": str(path),
                    "identity": actual,
                    "predicted_identity": "",
                    "status": "error",
                    "correct": False,
                })

            if i % 20 == 0:
                print(f"Processed {i}/{len(evaluation)}")

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_CSV, index=False)

    successful = result[result["status"] == "success"]
    detected = len(successful)
    total = len(result)

    print("\n--- Evaluation Summary ---")
    print("Evaluation images:", total)
    print("Faces detected:", detected)
    print("Detection coverage:", round(detected / total, 4) if total else 0)
    print("Correct predictions:", int(successful["correct"].sum()))
    print(
        "Accuracy on detected faces:",
        round(accuracy_score(y_true, y_pred), 4) if y_true else "N/A",
    )

    if y_true:
        print("\nClassification report (detected faces only):")
        print(classification_report(
            y_true,
            y_pred,
            labels=list(label_map.keys()),
            zero_division=0,
        ))

    print("\nSaved predictions:", OUTPUT_CSV)
    print("This script does not train or update the model.")


if __name__ == "__main__":
    main()

