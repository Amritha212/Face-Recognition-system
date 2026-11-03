
# Face Recognition System Using FaceNet

## Project Overview

This project evaluates a pre-trained FaceNet model and a partially fine-tuned FaceNet model for face recognition and verification. It uses MTCNN for face detection, FaceNet embeddings, and cosine similarity for comparing face images.

The project also includes a Flask dashboard for displaying dataset analytics and project results.

## Main Features

- Dataset organization and split analysis
- Face detection and embedding extraction
- Pre-trained FaceNet baseline evaluation
- Partial fine-tuning on training images
- Cosine-similarity verification
- ROC and precision-recall evaluation
- Similarity distribution plots
- Flask-based project dashboard

## Technologies

- Python 3.10
- PyTorch
- facenet-pytorch
- MTCNN
- OpenCV
- pandas
- scikit-learn
- Matplotlib
- Flask

## Project Structure

```text
Face_Recognition_project/
├── app.py
├── README.md
├── dataset/
├── models/
├── report/
│   ├── figures/
│   └── JEZT_Face_Recognition_Report.pdf
├── results/
├── src/
├── static/
├── templates/
└── venv/
```

## Setup

Open PowerShell in the project directory.

Install the dependencies used by the project:

```powershell
.\venv\Scripts\python.exe -m pip install torch facenet-pytorch pandas scikit-learn matplotlib flask pillow reportlab
```

## Run the Dashboard

```powershell
.\venv\Scripts\python.exe app.py
```

Open the local URL printed in the terminal, usually:

`http://127.0.0.1:5000`

## Run Experiments

Run these commands from the project root.

Baseline evaluation:

```powershell
.\venv\Scripts\python.exe src\baseline_evaluation.py
```

Partial fine-tuning:

```powershell
.\venv\Scripts\python.exe src\finetune_facenet.py
```

Pre-trained versus fine-tuned verification comparison:

```powershell
.\venv\Scripts\python.exe src\compare_verification_models.py
```

Generate baseline verification plots:

```powershell
.\venv\Scripts\python.exe src\analyze_baseline_verification.py
```

## Experimental Results

The high-quality verification experiment evaluated 236 images with valid embeddings from both models.

| Metric | Pre-trained FaceNet | Fine-tuned FaceNet |
|---|---:|---:|
| ROC-AUC | 0.980838 | 0.961675 |
| Average precision | 0.958213 | 0.900969 |
| Pair comparisons | 27,730 | 27,730 |

In this experiment, the pre-trained model performed better on both verification metrics. Fine-tuning did not improve the measured results.

These results are preliminary and apply to the selected high-quality evaluation subset. They do not establish performance on low-quality images or unseen identities.

## Important Limitations

- Four selected evaluation images failed face detection.
- Cross-quality near-duplicate leakage has not been fully ruled out.
- Pairwise comparisons share images and are not statistically independent.
- The exploratory baseline uses a different, smaller sample.
- Results should not be interpreted as proof of production readiness.

## Privacy and Responsible Use

Keep the supplied dataset private and local. Do not redistribute dataset images or use them for purposes outside the dataset provider's instructions. Face recognition results should not be treated as definitive identity verification in consequential settings.

## Report

See `report/JEZT_Face_Recognition_Report.pdf` for the methodology, results, figures, limitations, and evaluation-set declaration.
