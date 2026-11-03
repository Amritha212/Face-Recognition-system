
from pathlib import Path
import json
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv"
OUTPUT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

DEVICE = torch.device("cpu")
BATCH_SIZE = 8
EPOCHS = 5
MAX_IMAGES_PER_IDENTITY = 30


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    train_df = df[df["candidate_split"].str.lower() == "train"].copy()

    # Use high-quality images for the first controlled experiment.
    train_df = train_df[
        train_df["quality"].astype(str).str.lower() == "high_quality"
    ]

    train_df = (
        train_df.groupby("identity", group_keys=False)
        .head(MAX_IMAGES_PER_IDENTITY)
        .reset_index(drop=True)
    )

    identities = sorted(train_df["identity"].unique())
    label_map = {name: index for index, name in enumerate(identities)}

    print("Training images selected:", len(train_df))
    print("Identities:", label_map)
    print("Device:", DEVICE)

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )

    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

    embeddings = []
    labels = []
    skipped = []

    with torch.no_grad():
        for i, row in enumerate(train_df.itertuples(index=False), start=1):
            path = Path(row.image_path)
            try:
                with Image.open(path) as source:
                    face = detector(source.convert("RGB"))

                if face is None:
                    skipped.append(str(path))
                    continue

                embedding = model(face.unsqueeze(0).to(DEVICE))
                embedding = nn.functional.normalize(embedding, p=2, dim=1)

                embeddings.append(embedding.cpu().squeeze(0))
                labels.append(label_map[row.identity])

            except Exception as exc:
                print(f"Skipped {path.name}: {exc}")
                skipped.append(str(path))

            if i % 20 == 0:
                print(f"Processed {i}/{len(train_df)}")

    if not embeddings:
        raise RuntimeError("No embeddings were generated.")

    x = torch.stack(embeddings)
    y = torch.tensor(labels, dtype=torch.long)

    print("Usable training embeddings:", len(x))
    print("Skipped images:", len(skipped))

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    classifier = nn.Linear(x.shape[1], len(identities)).to(DEVICE)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        classifier.train()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = classifier(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_seen += len(batch_y)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={total_loss / total_seen:.4f} | "
            f"training_accuracy={total_correct / total_seen:.4f}"
        )

    model_path = MODEL_DIR / "identity_classifier.pt"
    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "label_map": label_map,
            "embedding_dimension": int(x.shape[1]),
            "pretrained_backbone": "InceptionResnetV1 VGGFace2",
            "training_note": "Frozen pretrained embeddings; classifier only.",
        },
        model_path,
    )

    with open(OUTPUT_DIR / "identity_label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    pd.DataFrame({
        "image_path": [str(p) for p in skipped]
    }).to_csv(OUTPUT_DIR / "classifier_skipped_images.csv", index=False)

    print("Saved classifier:", model_path)
    print("Saved label map:", OUTPUT_DIR / "identity_label_map.json")
    print("Note: FaceNet backbone was not fine-tuned.")


if __name__ == "__main__":
    main()

