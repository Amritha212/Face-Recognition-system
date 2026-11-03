from pathlib import Path
import json
import pandas as pd
from flask import Flask, render_template, jsonify

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# The dashboard expects this folder layout beside app.py:
# results/dataset_split_clean.csv (preferred) or results/dataset_split.csv
CSV_CANDIDATES = [
    BASE_DIR / "results" / "dataset_split_clean.csv",
    BASE_DIR / "results" / "dataset_split.csv",
]

def load_model_results():
    """Read saved verification results without rerunning model training."""
    summary_path = (
        BASE_DIR / "results" / "pre_post_verification_summary.csv"
    )

    if not summary_path.exists():
        return None

    try:
        df = pd.read_csv(summary_path)
        required = {
            "model",
            "valid_images",
            "pair_count",
            "roc_auc",
            "average_precision",
        }

        if not required.issubset(df.columns):
            return None

        return df.to_dict(orient="records")

    except (OSError, ValueError, pd.errors.ParserError):
        return None



def load_dataset():
    csv_path = next((p for p in CSV_CANDIDATES if p.exists()), None)
    if csv_path is None:
        return None, None, "Dataset split CSV not found in the results folder."
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        return None, csv_path.name, f"Could not read CSV: {exc}"

    required = {"image_path", "identity", "quality", "split"}
    missing_columns = required - set(df.columns)
    if missing_columns:
        return None, csv_path.name, "Missing columns: " + ", ".join(sorted(missing_columns))
    df = df.dropna(subset=["identity", "split"]).copy()
    return df, csv_path.name, None


def build_payload():
    df, csv_name, error = load_dataset()
    model_results = load_model_results()
    if error:
        model_results = load_model_results()
        return {"error": error, "csv_name": csv_name, "ready": False}

    # Check paths without exposing or serving the images themselves.
    missing_paths = 0
    for path in df["image_path"].dropna().astype(str):
        if not Path(path).is_file():
            missing_paths += 1

    identity_counts = df.groupby("identity").size().sort_values(ascending=False)
    quality_counts = df["quality"].fillna("Unknown").value_counts()
    split_counts = df["split"].astype(str).str.lower().value_counts()

    # Normalize common naming variations for the split cards.
    train_count = int(sum(v for k, v in split_counts.items() if "train" in k and "eval" not in k and "test" not in k))
    eval_count = int(sum(v for k, v in split_counts.items() if "eval" in k or "test" in k))

    return {
        "ready": True,
        "csv_name": csv_name,
        "total": int(len(df)),
        "identity_total": int(df["identity"].nunique()),
        "train_count": train_count,
        "eval_count": eval_count,
        "missing_paths": int(missing_paths),
        "identities": [{"name": str(k), "count": int(v)} for k, v in identity_counts.items()],
        "quality": [{"name": str(k), "count": int(v)} for k, v in quality_counts.items()],
        "split": [{"name": str(k), "count": int(v)} for k, v in split_counts.items()],
        "dataset_path": "dataset/original/",
        "model_status": (
            "FaceNet pretrained and fine-tuned models evaluated"
            if model_results else "Saved evaluation results not found"
        ),
        "evaluation_status": (
            "Verification results available"
            if model_results else "Not run"
        ),
        "duplicate_status": "Review required",
        "model_results": model_results or [],
    }


@app.route("/")
def dashboard():
    data = build_payload()
    return render_template("index.html", data=data)


@app.route("/api/dataset")
def dataset_api():
    return jsonify(build_payload())


@app.route("/api/analytics")
def analytics_api():
    df, csv_name, error = load_dataset()

    if error:
        return jsonify({
            "ready": False,
            "error": error
        }), 500

    # Identity-wise image counts
    identity_counts = (
        df.groupby("identity")
        .size()
        .sort_values(ascending=False)
    )

    # Image-quality distribution
    quality_counts = (
        df["quality"]
        .fillna("Unknown")
        .astype(str)
        .value_counts()
    )

    # Training/evaluation distribution
    split_counts = (
        df["split"]
        .fillna("Unknown")
        .astype(str)
        .str.lower()
        .value_counts()
    )

    # Quality by identity
    quality_by_identity = (
        df.assign(quality=df["quality"].fillna("Unknown").astype(str))
        .groupby(["identity", "quality"])
        .size()
        .unstack(fill_value=0)
    )

    return jsonify({
        "ready": True,
        "csv_name": csv_name,
        "total_images": int(len(df)),
        "identity_labels": [str(x) for x in identity_counts.index],
        "identity_counts": [int(x) for x in identity_counts.values],
        "quality_labels": [str(x) for x in quality_counts.index],
        "quality_counts": [int(x) for x in quality_counts.values],
        "split_labels": [str(x) for x in split_counts.index],
        "split_counts": [int(x) for x in split_counts.values],
        "quality_by_identity": {
            str(identity): {
                str(quality): int(count)
                for quality, count in row.items()
            }
            for identity, row in quality_by_identity.iterrows()
        }
    })



if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
