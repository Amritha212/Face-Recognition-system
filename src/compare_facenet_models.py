
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv"
FINETUNED_PATH = BASE_DIR / "models" / "facenet_finetuned_partial.pt"
PRE_CLASSIFIER_PATH = BASE_DIR / "models" / "identity_classifier.pt"
FINE_CLASSIFIER_PATH = BASE_DIR / "models" / "finetuned_matched_classifier.pt"
OUTPUT_PATH = BASE_DIR / "results" / "pre_post_comparison.csv"

DEVICE = torch.device("cpu")
MAX_PER_IDENTITY = 30


def load_classifier(path, device):
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )
    label_map = checkpoint["label_map"]
    classifier = nn.Linear(
        checkpoint["embedding_dimension"], len(label_map)
    ).to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()
    return classifier, label_map


def main():
    df = pd.read_csv(CSV_PATH)

    evaluation = df[
        df["candidate_split"].astype(str).str.lower().eq("evaluation")
        & df["quality"].astype(str).str.lower().eq("high_quality")
    ].copy()

    evaluation = (
        evaluation.groupby("identity", group_keys=False)
        .head(MAX_PER_IDENTITY)
        .reset_index(drop=True)
    )

    fine_checkpoint = torch.load(
        FINETUNED_PATH, map_location=DEVICE, weights_only=False
    )

    pretrained = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    finetuned = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    finetuned.load_state_dict(fine_checkpoint["model_state_dict"])
    finetuned.eval()

    pretrained_classifier, pre_labels = load_classifier(
        PRE_CLASSIFIER_PATH, DEVICE
    )
    finetuned_classifier, fine_labels = load_classifier(
        FINE_CLASSIFIER_PATH, DEVICE
    )

    if pre_labels != fine_labels:
        raise ValueError("Pretrained and fine-tuned classifier label maps differ.")

    label_map = fine_labels
    reverse_map = {value: key for key, value in label_map.items()}

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
        for i, row in enumerate(evaluation.itertuples(index=False), start=1):
            path = Path(row.image_path)
            actual = str(row.identity)

            record = {
                "image_path": str(path),
                "identity": actual,
                "quality": row.quality,
                "status": "",
                "pretrained_prediction": "",
                "finetuned_prediction": "",
            }

            try:
                with Image.open(path) as source:
                    face = detector(source.convert("RGB"))

                if face is None:
                    record["status"] = "no_face_detected"
                    records.append(record)
                    continue

                face = face.unsqueeze(0).to(DEVICE)

                pre_embedding = nn.functional.normalize(
                    pretrained(face), p=2, dim=1
                )
                pre_prediction = reverse_map[
                    int(pretrained_classifier(pre_embedding).argmax(1).item())
                ]

                fine_embedding = nn.functional.normalize(
                    finetuned(face), p=2, dim=1
                )
                fine_prediction = reverse_map[
                    int(finetuned_classifier(fine_embedding).argmax(1).item())
                ]

                record["status"] = "success"
                record["pretrained_prediction"] = pre_prediction
                record["finetuned_prediction"] = fine_prediction

            except Exception as exc:
                print(f"Error on {path.name}: {exc}")
                record["status"] = "error"

            records.append(record)

            if i % 20 == 0:
                print(f"Processed {i}/{len(evaluation)}")

    result = pd.DataFrame(records)
    valid = result[result["status"].eq("success")].copy()

    result["pretrained_correct"] = (
        result["pretrained_prediction"] == result["identity"]
    ) & result["status"].eq("success")
    result["finetuned_correct"] = (
        result["finetuned_prediction"] == result["identity"]
    ) & result["status"].eq("success")

    result.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Pretrained vs Fine-tuned ---")
    print("Evaluation images:", len(result))
    print("Faces detected:", len(valid))
    print("Detection failures:", int((result["status"] == "no_face_detected").sum()))

    if len(valid):
        print(
            "Pretrained accuracy:",
            round(accuracy_score(
                valid["identity"], valid["pretrained_prediction"]
            ), 4),
        )
        print(
            "Fine-tuned accuracy:",
            round(accuracy_score(
                valid["identity"], valid["finetuned_prediction"]
            ), 4),
        )

        print("\nFine-tuned classification report:")
        print(classification_report(
            valid["identity"],
            valid["finetuned_prediction"],
            labels=list(label_map.keys()),
            zero_division=0,
        ))

    print("\nSaved:", OUTPUT_PATH)
    print("Evaluation images were not used for training.")


if __name__ == "__main__":
    main()

