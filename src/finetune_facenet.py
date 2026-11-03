
from pathlib import Path
import json
import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "grouped_split_candidate.csv"
MODEL_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DEVICE = torch.device("cpu")
RANDOM_SEED = 42
MAX_IMAGES_PER_IDENTITY = 20
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 0.00001


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Training split only. Evaluation images are never loaded here.
    train_df = df[
        df["candidate_split"].astype(str).str.lower().eq("train")
        & df["quality"].astype(str).str.lower().eq("high_quality")
    ].copy()

    train_df = (
        train_df.groupby("identity", group_keys=False)
        .sample(
            n=MAX_IMAGES_PER_IDENTITY,
            random_state=RANDOM_SEED,
            replace=False,
        )
        .reset_index(drop=True)
    )

    identities = sorted(train_df["identity"].unique())
    label_map = {name: index for index, name in enumerate(identities)}

    print("Training images selected:", len(train_df))
    print("Identity labels:", label_map)
    print("Device:", DEVICE)

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )

    model = InceptionResnetV1(pretrained="vggface2").to(DEVICE)

    # Freeze all parameters first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune the final embedding layers.
    for parameter in model.last_linear.parameters():
        parameter.requires_grad = True
    for parameter in model.last_bn.parameters():
        parameter.requires_grad = True

    # Extract face crops before training to reduce repeated detection cost.
    face_tensors = []
    labels = []
    skipped = []

    model.eval()
    for i, row in enumerate(train_df.itertuples(index=False), start=1):
        path = Path(row.image_path)

        try:
            with Image.open(path) as source:
                face = detector(source.convert("RGB"))

            if face is None:
                skipped.append(str(path))
                continue

            face_tensors.append(face)
            labels.append(label_map[row.identity])

        except Exception as exc:
            print(f"Skipped {path.name}: {exc}")
            skipped.append(str(path))

        if i % 20 == 0:
            print(f"Detected faces: processed {i}/{len(train_df)}")

    if not face_tensors:
        raise RuntimeError("No training faces detected.")

    x = torch.stack(face_tensors)
    y = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    classifier = nn.Linear(512, len(identities)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    trainable_parameters = [
        p for p in model.parameters() if p.requires_grad
    ] + list(classifier.parameters())

    optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    history = []

    for epoch in range(EPOCHS):
        model.train()
        classifier.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            embeddings = model(batch_x)
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
            logits = classifier(embeddings)

            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            total_correct += (
                logits.argmax(dim=1) == batch_y
            ).sum().item()
            total_seen += len(batch_y)

        epoch_loss = total_loss / total_seen
        epoch_accuracy = total_correct / total_seen

        history.append({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "training_accuracy": epoch_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={epoch_loss:.4f} | "
            f"training_accuracy={epoch_accuracy:.4f}"
        )

    checkpoint_path = MODEL_DIR / "facenet_finetuned_partial.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "classifier_state_dict": classifier.state_dict(),
        "label_map": label_map,
        "embedding_dimension": 512,
        "pretrained_weights": "vggface2",
        "fine_tuned_layers": ["last_linear", "last_bn"],
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "random_seed": RANDOM_SEED,
        "training_images_selected": len(train_df),
        "training_faces_used": len(x),
        "training_images_skipped": skipped,
    }, checkpoint_path)

    pd.DataFrame(history).to_csv(
        RESULTS_DIR / "finetune_training_history.csv",
        index=False,
    )

    with open(
        RESULTS_DIR / "finetune_label_map.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(label_map, f, indent=2)

    print("\nSaved fine-tuned checkpoint:", checkpoint_path)
    print("Saved training history:", RESULTS_DIR / "finetune_training_history.csv")
    print("Usable training faces:", len(x))
    print("Skipped images:", len(skipped))
    print("Only the training split was used.")


if __name__ == "__main__":
    main()
