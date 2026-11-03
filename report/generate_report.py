
from pathlib import Path

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    PageBreak, Table, TableStyle, Preformatted
)

BASE = Path(__file__).resolve().parent.parent
REPORT_DIR = BASE / "report"
FIGURES = REPORT_DIR / "figures"
RESULTS = BASE / "results"
PDF_PATH = REPORT_DIR / "JEZT_Face_Recognition_Report.pdf"

summary_path = RESULTS / "pre_post_verification_summary.csv"
summary = pd.read_csv(summary_path)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="CoverTitle", parent=styles["Title"],
    fontSize=23, leading=30, alignment=TA_CENTER,
    textColor=colors.HexColor("#17365D"), spaceAfter=18
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading1"],
    fontSize=16, leading=20, textColor=colors.HexColor("#17365D"),
    spaceBefore=8, spaceAfter=10
))
styles.add(ParagraphStyle(
    name="SubHeading", parent=styles["Heading2"],
    fontSize=12, leading=15, textColor=colors.HexColor("#24527A"),
    spaceBefore=6, spaceAfter=6
))
styles.add(ParagraphStyle(
    name="BodyCustom", parent=styles["BodyText"],
    fontSize=9.5, leading=14, spaceAfter=7
))
styles.add(ParagraphStyle(
    name="SmallCustom", parent=styles["BodyText"],
    fontSize=8, leading=11, spaceAfter=4
))

doc = SimpleDocTemplate(
    str(PDF_PATH), pagesize=A4,
    rightMargin=1.8 * cm, leftMargin=1.8 * cm,
    topMargin=1.7 * cm, bottomMargin=1.7 * cm,
)

story = []


def para(text, style="BodyCustom"):
    story.append(Paragraph(text, styles[style]))


def heading(text):
    para(text, "SectionHeading")


def image_file(filename, width=16 * cm, height=9 * cm):
    path = FIGURES / filename
    if path.exists():
        img = Image(str(path))
        scale = min(width / img.imageWidth, height / img.imageHeight)
        img.drawWidth = img.imageWidth * scale
        img.drawHeight = img.imageHeight * scale
        story.append(img)
    else:
        para(f"Figure not found: {filename}", "SmallCustom")


def footer(canvas, document):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(1.8 * cm, 0.9 * cm, "Face Recognition Research Project")
    canvas.drawRightString(19.2 * cm, 0.9 * cm, f"Page {document.page}")
    canvas.restoreState()


# Extract measured values from the generated summary CSV.
pre = summary.loc[summary["model"] == "pretrained"].iloc[0]
fine = summary.loc[summary["model"] == "finetuned"].iloc[0]

# PAGE 1 — Cover and abstract
story.append(Spacer(1, 2.2 * cm))
para("FACE RECOGNITION SYSTEM", "CoverTitle")
para("Pre-trained and Partially Fine-tuned FaceNet Evaluation", "Heading2")
story.append(Spacer(1, 1 * cm))
para("<b>Technical Project Report</b>", "BodyCustom")
para("Prepared for: JEZT Technologies Internship Assignment", "BodyCustom")
para("Evaluation date: 28 October 2026", "BodyCustom")
story.append(Spacer(1, 1.2 * cm))
heading("Abstract")
para(
    "This project investigates face recognition and verification using a "
    "pre-trained FaceNet model and a partially fine-tuned version. The workflow "
    "includes dataset preparation, train/evaluation partitioning, face detection, "
    "embedding extraction, cosine-similarity scoring, and evaluation using "
    "ROC-AUC and average precision. A small exploratory baseline and a separate "
    "high-quality evaluation experiment were conducted. The high-quality "
    "verification experiment used the same successfully detected images for both "
    "models. The pre-trained model achieved ROC-AUC "
    f"{pre['roc_auc']:.4f} and average precision {pre['average_precision']:.4f}; "
    "the fine-tuned model achieved ROC-AUC "
    f"{fine['roc_auc']:.4f} and average precision "
    f"{fine['average_precision']:.4f}. Fine-tuning did not improve these metrics "
    "in this experiment. Results are preliminary and limitations include "
    "incomplete cross-quality leakage verification and limited evaluation of "
    "low-quality images."
)
story.append(Spacer(1, 1 * cm))
para(
    "<b>Privacy:</b> The supplied dataset is intended to remain local and private. "
    "This report does not reproduce or distribute dataset images.",
    "BodyCustom"
)
story.append(PageBreak())

# PAGE 2 — Dataset and split
heading("1. Dataset and Data Preparation")
para(
    "The project dataset contains eight identity folders: Ananthu, Firoz, "
    "Mujeeb, Ruvais, Sebin, Shinjil, Suresh, and Thomas. The source collection "
    "contains images in high-quality and low-quality folders. Image quality "
    "varies substantially, including very small images where reliable face "
    "detection may be difficult."
)
para(
    "The dataset preparation workflow records image paths, identity labels, "
    "quality categories, and split assignments. A candidate grouped split was "
    "created for evaluation. High-quality images were grouped by identity and "
    "frame-number blocks to reduce the risk of placing adjacent frames from the "
    "same short sequence in both partitions."
)
para(
    "The candidate split contains 17,667 training rows and 4,465 evaluation "
    "rows. A SHA-256 audit found no exact duplicate file hashes crossing the "
    "candidate split, and the implemented high-quality frame-group check found "
    "no frame groups represented in both partitions under its grouping rule."
)
para(
    "<b>Important caveat:</b> These checks do not prove that the full dataset is "
    "free from near-duplicate leakage. Cross-quality relationships have not been "
    "fully ruled out. The split should therefore be described as a candidate "
    "split, not as conclusively leakage-free."
)
para("Data files used:", "SubHeading")
for item in [
    "results/dataset_split.csv",
    "results/dataset_split_clean.csv",
    "results/grouped_split_candidate.csv",
    "results/near_duplicate_audit.csv",
    "results/suspected_near_duplicates.csv",
]:
    para("• " + item, "SmallCustom")
para(
    "Dataset images are not included in this report. All processing should "
    "follow the dataset provider's privacy and usage requirements."
)
story.append(PageBreak())

# PAGE 3 — Model and methods
heading("2. Model and Experimental Method")
para(
    "The recognition backbone is FaceNet's InceptionResnetV1 implementation "
    "with VGGFace2 pre-trained weights. MTCNN is used to detect and crop a face "
    "before the crop is passed to the embedding network. Each embedding is "
    "L2-normalized, and cosine similarity is used to score image pairs."
)
para("Baseline verification", "SubHeading")
para(
    "An exploratory baseline used an 80-image sample. It produced 27 valid "
    "embeddings and 351 pair comparisons. Its ROC-AUC was 0.7869 and average "
    "precision was 0.7112. Because many low-quality images failed face detection, "
    "this baseline is limited and should not be directly compared with the "
    "larger high-quality evaluation experiment."
)
para("Partial fine-tuning", "SubHeading")
para(
    "A partial fine-tuning experiment selected 160 high-quality training images "
    "(20 per identity). All 160 face crops were detected. The last linear layer "
    "and last batch-normalization layer were unfrozen, and training ran for "
    "three epochs with batch size 8 and learning rate 0.00001. The checkpoint "
    "was saved as models/facenet_finetuned_partial.pt."
)
para(
    "A separate identity classifier was trained on 240 high-quality training "
    "embeddings for each pipeline. These classifiers were used for the separate "
    "identity-classification experiment. The verification experiment reported "
    "in this document compares embedding similarity scores, not classifier "
    "predictions."
)
para("Verification evaluation", "SubHeading")
para(
    "The comparison selected up to 30 high-quality evaluation images per "
    "identity. Of 240 selected images, 236 produced valid embeddings for both "
    "models and four failed face detection. Pairwise scores were computed on "
    "the same 236 images for both models, resulting in 27,730 pairs: 3,364 "
    "same-identity pairs and 24,366 different-identity pairs."
)
para(
    "ROC-AUC measures ranking discrimination across thresholds. Average "
    "precision summarizes the precision-recall curve. Pair scores share images "
    "and therefore are not statistically independent observations."
)
story.append(PageBreak())

# PAGE 4 — Results
heading("3. Verification Results")
table_data = [
    ["Metric", "Pre-trained FaceNet", "Fine-tuned FaceNet"],
    ["Valid images", str(int(pre["valid_images"])), str(int(fine["valid_images"]))],
    ["Pair comparisons", str(int(pre["pair_count"])), str(int(fine["pair_count"]))],
    ["Same-identity pairs", str(int(pre["same_identity_pairs"])),
     str(int(fine["same_identity_pairs"]))],
    ["Different-identity pairs", str(int(pre["different_identity_pairs"])),
     str(int(fine["different_identity_pairs"]))],
    ["ROC-AUC", f"{pre['roc_auc']:.6f}", f"{fine['roc_auc']:.6f}"],
    ["Average precision", f"{pre['average_precision']:.6f}",
     f"{fine['average_precision']:.6f}"],
]
table = Table(table_data, colWidths=[6.2 * cm, 5.2 * cm, 5.2 * cm])
table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
    ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F3F6FA")),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story.append(table)
story.append(Spacer(1, 0.5 * cm))
para("ROC curve", "SubHeading")
image_file("pre_post_verification_roc.png", height=7 * cm)
para("Precision-recall curve", "SubHeading")
image_file("pre_post_verification_pr.png", height=7 * cm)
story.append(PageBreak())

# PAGE 5 — Similarity distribution and discussion
heading("4. Similarity Analysis and Discussion")
para("Same-identity and different-identity similarity distributions", "SubHeading")
image_file("pre_post_verification_similarity.png", height=9.5 * cm)
para(
    "The pre-trained model achieved ROC-AUC 0.9808 and average precision "
    "0.9582. The fine-tuned model achieved ROC-AUC 0.9617 and average precision "
    "0.9010. In this experiment, the pre-trained model performed better on both "
    "verification metrics. Fine-tuning therefore did not demonstrate an "
    "improvement under the tested configuration."
)
para(
    "Possible factors include the small fine-tuning sample, limited training "
    "duration, classifier/backbone interactions, and the characteristics of "
    "the selected high-quality subset. These are hypotheses, not confirmed "
    "causes. Further controlled experiments would be required to determine why "
    "the fine-tuned pipeline performed worse."
)
para("Limitations", "SubHeading")
for item in [
    "The main verification comparison uses high-quality images and does not establish low-quality performance.",
    "Four selected images failed face detection and were excluded from pairwise metrics.",
    "The candidate split has not been fully verified for cross-quality near-duplicate leakage.",
    "Pairwise comparisons share images; confidence intervals should use identity- or image-aware resampling rather than treating all pairs as independent.",
    "The exploratory baseline uses a different, much smaller sample and should not be compared directly with the 236-image verification experiment.",
    "No claim is made that the system is suitable for production, security, or consequential identity decisions.",
]:
    para("• " + item, "SmallCustom")
story.append(PageBreak())

# PAGE 6 — Reproducibility and declaration
heading("5. Reproducibility and Deliverables")
para("Environment", "SubHeading")
para(
    "The experiments were run on Windows using Python 3.10 and CPU-only "
    "PyTorch. The virtual environment is stored in the project folder. "
    "Commands below assume PowerShell is opened at the project root."
)
commands = (
    r".\venv\Scripts\python.exe src\baseline_evaluation.py" "\n"
    r".\venv\Scripts\python.exe src\finetune_facenet.py" "\n"
    r".\venv\Scripts\python.exe src\compare_verification_models.py"
)
story.append(Preformatted(commands, styles["Code"]))
para(
    "The baseline analysis and verification comparison also generate CSV "
    "summaries and PNG figures. Paths and dependencies should be checked before "
    "re-running scripts. Model weights and source dataset files are local "
    "artifacts and should not be distributed unless permission is granted."
)
para("Key source and output files", "SubHeading")
for item in [
    "src/baseline_evaluation.py",
    "src/finetune_facenet.py",
    "src/compare_verification_models.py",
    "results/pre_post_verification_pair_scores.csv",
    "results/pre_post_verification_summary.csv",
    "results/finetune_training_history.csv",
    "models/facenet_finetuned_partial.pt",
]:
    para("• " + item, "SmallCustom")
heading("6. Evaluation-Set Declaration")
para(
    "Declaration: I confirm that evaluation images were not used for gradient "
    "training in the reported verification experiment. The evaluation subset "
    "was used only for model evaluation. I acknowledge that the candidate split "
    "has not been fully verified for cross-quality near-duplicate leakage, as "
    "discussed in this report."
)
para(
    "Sign only if this statement accurately reflects your final workflow. "
    "If any evaluation images or their labels were used to train, fine-tune, "
    "select model parameters, or train the reported classifiers, revise this "
    "declaration before signing."
)
story.append(Spacer(1, 0.8 * cm))
para("Name: __________________________________________")
para("Signature: _______________________________________")
para("Date: ___________________________________________")
story.append(Spacer(1, 0.8 * cm))
para(
    "End of report. All metrics are experimental results from the stated "
    "samples and should be interpreted with the limitations documented above.",
    "SmallCustom"
)

doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(f"Report created: {PDF_PATH}")
print(f"Pre-trained ROC-AUC: {pre['roc_auc']:.6f}")
print(f"Fine-tuned ROC-AUC: {fine['roc_auc']:.6f}")
