
from pathlib import Path

import pandas as pd
import cv2

PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_DIR / "results" / "near_duplicate_audit.csv"
OUTPUT_IMAGE = PROJECT_DIR / "results" / "near_duplicate_review_sheet.jpg"

MAX_PAIRS = 20
THUMB_WIDTH = 180
THUMB_HEIGHT = 180
LABEL_HEIGHT = 55


def load_thumbnail(path):
    image = cv2.imread(str(path))
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (THUMB_WIDTH, THUMB_HEIGHT))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def main():
    df = pd.read_csv(INPUT_CSV)
    candidates = df[df["hash_distance"] == 0].head(MAX_PAIRS)

    if candidates.empty:
        print("No distance-0 pairs found.")
        return

    canvas_height = len(candidates) * (THUMB_HEIGHT + LABEL_HEIGHT)
    canvas_width = THUMB_WIDTH * 2 + 40

    canvas = 245 * __import__("numpy").ones(
        (canvas_height, canvas_width, 3), dtype="uint8"
    )

    for i, row in enumerate(candidates.itertuples(index=False)):
        y = i * (THUMB_HEIGHT + LABEL_HEIGHT)
        left = load_thumbnail(row.train_path)
        right = load_thumbnail(row.evaluation_path)

        if left is None or right is None:
            print(f"Could not read pair {i + 1}")
            continue

        canvas[y:y + THUMB_HEIGHT, 0:THUMB_WIDTH] = left
        canvas[y:y + THUMB_HEIGHT, THUMB_WIDTH + 40:THUMB_WIDTH * 2 + 40] = right

        label = (
            f"{row.identity} | hash distance: {row.hash_distance}"
        )
        cv2.putText(
            canvas, label, (5, y + THUMB_HEIGHT + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1
        )

        cv2.putText(
            canvas, "TRAIN", (5, y + THUMB_HEIGHT + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1
        )
        cv2.putText(
            canvas, "EVALUATION",
            (THUMB_WIDTH + 40, y + THUMB_HEIGHT + 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1
        )

    output = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_IMAGE), output)

    print(f"Pairs included: {len(candidates)}")
    print(f"Review sheet saved to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()
