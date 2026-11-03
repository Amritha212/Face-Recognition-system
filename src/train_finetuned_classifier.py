
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv"
CHECKPOINT_PATH = BASE_DIR / "models" / "facenet_finetuned_partial.pt"
OUTPUT_PATH = BASE_DIR / "models" / "finetuned_matched_classifier.pt"

DEVICE = torch.device("cpu")
MAX_IMAGES_PER_IDENTITY = 30
EPOCHS = 10
BATCH_SIZE = 8


def main():
    df = pd.read_csv(CSV_PATH)
    train_df = df[
        df["candidate_split"].astype(str).str.lower().eq("train")
        & df["quality"].astype(str).str.lower().eq("high_quality")
    ].copy()

    train_df = (
        train_df.groupby("identity", group_keys=False)
        .head(MAX_IMAGES_PER_IDENTITY)
        .reset_index(drop=True)
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH, map_location=DEVICE, weights_only=False
    )
    label_map = checkpoint["label_map"]

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )

    backbone = InceptionResnetV1(pretrained="vggface2").to(DEVICE)
    backbone.load_state_dict(checkpoint["model_state_dict"])
    backbone.eval()

    features = []
    labels = []

    with torch.no_grad():
        for i, row in enumerate(train_df.itertuples(index=False), start=1):
            try:
                with Image.open(row.image_path) as source:
                    face = detector(source.convert("RGB"))

                if face is None:
                    continue

                embedding = backbone(face.unsqueeze(0).to(DEVICE))
                embedding = nn.functional.normalize(embedding, p=2, dim=1)
                features.append(embedding.squeeze(0).cpu())
                labels.append(label_map[row.identity])

            except Exception as exc:
                print(f"Skipped {Path(row.image_path).name}: {exc}")

            if i % 20 == 0:
                print(f"Processed {i}/{len(train_df)}")

    if not features:
        raise RuntimeError("No training embeddings were generated.")

    x = torch.stack(features)
    y = torch.tensor(labels, dtype=torch.long)

    print("Training embeddings:", len(x))

    classifier = nn.Linear(512, len(label_map)).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(x, y), batch_size=BATCH_SIZE, shuffle=True
    )

    for epoch in range(EPOCHS):
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for bx, by in loader:
            optimizer.zero_grad()
            logits = classifier(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(by)
            total_correct += (logits.argmax(1) == by).sum().item()
            total_seen += len(by)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={total_loss / total_seen:.4f} | "
            f"training_accuracy={total_correct / total_seen:.4f}"
        )

    torch.save({
        "classifier_state_dict": classifier.state_dict(),
        "label_map": label_map,
        "embedding_dimension": 512,
        "backbone_checkpoint": str(CHECKPOINT_PATH),
        "training_embeddings": len(x),
    }, OUTPUT_PATH)

    print("Saved matched classifier:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
