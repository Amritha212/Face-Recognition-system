
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "dataset_split_clean.csv"
DEVICE = torch.device("cpu")

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df["split"] = df["split"].astype(str).str.lower()

    # Use evaluation images for this small, inference-only smoke test.
    eval_df = df[df["split"].str.contains("eval|test", regex=True)]

    if eval_df.empty:
        raise ValueError("No evaluation images found in the CSV.")

    # Select at most two images per identity to keep this test small.
    sample = (
        eval_df.groupby("identity", group_keys=False)
        .head(2)
        .reset_index(drop=True)
    )

    print(f"Device: {DEVICE}")
    print(f"Images selected for smoke test: {len(sample)}")
    print("Loading face detector and pretrained FaceNet model...")

    detector = MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        device=DEVICE
    )

    model = InceptionResnetV1(
        pretrained="vggface2"
    ).eval().to(DEVICE)

    results = []
    embedding_vectors = []

    for _, row in sample.iterrows():
        image_path = Path(str(row["image_path"]))

        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")

            face = detector(image)

            if face is None:
                print(f"NO FACE DETECTED: {image_path.name}")
                results.append({
                    "identity": row["identity"],
                    "image": image_path.name,
                    "status": "No face detected"
                })
                continue

            with torch.inference_mode():
                embedding = model(face.unsqueeze(0).to(DEVICE))

            vector = embedding.squeeze(0).cpu().numpy()

            if not np.isfinite(vector).all():
                raise ValueError("Embedding contains non-finite values.")

            embedding_vectors.append(vector)
            results.append({
                "identity": row["identity"],
                "image": image_path.name,
                "status": "Success",
                "embedding_dimensions": len(vector),
                "embedding_norm": float(np.linalg.norm(vector))
            })

            print(
                f"SUCCESS: {image_path.name} | "
                f"embedding shape: {tuple(embedding.shape)}"
            )

        except Exception as exc:
            print(f"ERROR: {image_path.name}: {exc}")
            results.append({
                "identity": row["identity"],
                "image": image_path.name,
                "status": f"Error: {exc}"
            })

    output_path = BASE_DIR / "results" / "facenet_smoke_test.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)

    print("\n--- Smoke Test Summary ---")
    print(f"Images selected: {len(sample)}")
    print(f"Valid embeddings: {len(embedding_vectors)}")
    print(f"Results saved to: {output_path}")

    if embedding_vectors:
        print(
            "Embedding dimensions:",
            embedding_vectors[0].shape[0]
        )

    if not embedding_vectors:
        print("No valid embeddings were generated. Check the errors above.")


if __name__ == "__main__":
    main()
