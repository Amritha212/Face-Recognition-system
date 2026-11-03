
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_DIR = Path(__file__).resolve().parent.parent
PAIRS_CSV = PROJECT_DIR / "results" / "suspected_near_duplicates.csv"
OUTPUT_IMAGE = PROJECT_DIR / "results" / "duplicate_review.jpg"

pairs = pd.read_csv(PAIRS_CSV)
pairs = pairs[pairs["distance"] <= 1].head(9)

if pairs.empty:
    print("No pairs with hash distance 0 or 1.")
    raise SystemExit

cards = []
for _, row in pairs.iterrows():
    try:
        path_a = Path(row["path_a"])
        path_b = Path(row["path_b"])

        with Image.open(path_a) as im:
            a = im.convert("RGB")
            a.thumbnail((220, 180))

        with Image.open(path_b) as im:
            b = im.convert("RGB")
            b.thumbnail((220, 180))

        card = Image.new("RGB", (460, 260), "white")
        card.paste(a, (5, 30))
        card.paste(b, (235, 30))

        draw = ImageDraw.Draw(card)
        draw.text(
            (5, 5),
            f"Hash distance: {row['distance']}",
            fill="black",
        )
        draw.text((5, 215), path_a.name[:35], fill="black")
        draw.text((235, 215), path_b.name[:35], fill="black")
        draw.text(
            (5, 238),
            f"{row['identity_a']} ({row['split_a']}) vs "
            f"{row['identity_b']} ({row['split_b']})",
            fill="black",
        )
        cards.append(card)
    except Exception as exc:
        print("Could not inspect pair:", exc)

if cards:
    sheet = Image.new("RGB", (920, 260 * len(cards)), "#dddddd")
    for i, card in enumerate(cards):
        sheet.paste(card, (0, i * 260))

    sheet.save(OUTPUT_IMAGE, quality=90)
    print("Review image saved to:", OUTPUT_IMAGE)
    print("Pairs displayed:", len(cards))
