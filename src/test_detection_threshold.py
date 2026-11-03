
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from facenet_pytorch import MTCNN

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "results" / "baseline_embedding_status.csv"
OUTPUT_PATH = BASE_DIR / "results" / "detection_comparison.csv"

DEVICE = torch.device("cpu")


def make_detector():
    return MTCNN(
        image_size=160,
        margin=20,
        keep_all=False,
        thresholds=[0.5, 0.6, 0.6],
        min_face_size=5,
        device=DEVICE,
    )


def detect(image, detector):
    try:
        return detector(image) is not None
    except Exception as exc:
        print("Detection error:", exc)
        return False


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    if "image_path" not in df.columns:
        raise ValueError("CSV does not contain image_path.")

    df = df.drop_duplicates(subset=["image_path"]).copy()

    print("Images to compare:", len(df))
    print("Device:", DEVICE)
    print("Loading detector...")

    detector = make_detector()
    results = []

    for index, row in df.iterrows():
        path = Path(str(row["image_path"]))
        print(f"[{len(results) + 1}/{len(df)}] {path.name}")

        original_detected = False
        upscaled_detected = False
        error_message = ""

        try:
            with Image.open(path) as source:
                image = source.convert("RGB")
                original_size = image.size

                # Condition A: original image
                original_detected = detect(image, detector)

                # Condition B: same image enlarged 4x
                enlarged = image.resize(
                    (image.width * 4, image.height * 4),
                    Image.Resampling.LANCZOS,
                )
                upscaled_detected = detect(enlarged, detector)

        except Exception as exc:
            error_message = str(exc)
            print("Image error:", error_message)
            original_size = None

        results.append({
            "image_path": str(path),
            "identity": row.get("identity", ""),
            "quality": row.get("quality", ""),
            "original_width": original_size[0] if original_size else None,
            "original_height": original_size[1] if original_size else None,
            "original_detected": original_detected,
            "upscaled_4x_detected": upscaled_detected,
            "error": error_message,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_PATH, index=False)

    print("\n--- Detection Comparison ---")
    print("Images tested:", len(result_df))

    for column in ["original_detected", "upscaled_4x_detected"]:
        count = int(result_df[column].sum())
        print(f"{column}: {count}/{len(result_df)}")

    print("\nDetection counts by quality:")
    print(
        result_df.groupby("quality")[
            ["original_detected", "upscaled_4x_detected"]
        ].sum().to_string()
    )

    print("\nSaved:", OUTPUT_PATH)
    print("This test changes no original images or baseline files.")


if __name__ == "__main__":
    main()
