# Face Recognition and Verification System

A Python-based Face Recognition and Verification System developed using FaceNet, MTCNN, PyTorch, OpenCV, and Flask.

This project explores face detection, feature extraction, identity classification, and face verification through pretrained and partially fine-tuned deep learning models.

## Project Overview

The system uses deep learning techniques to detect and recognize faces, extract facial embeddings, and compare facial similarities. It also includes a Flask dashboard to present model evaluation results and visualizations.

## Key Features

- Face detection using MTCNN.
- Facial feature extraction using FaceNet.
- Face recognition and identity classification.
- Face verification using facial similarity scores.
- Pretrained and partially fine-tuned model comparison.
- Dataset quality and duplicate analysis.
- Evaluation using ROC-AUC and Average Precision.
- Interactive Flask dashboard for displaying results.

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Face Detection | MTCNN |
| Face Recognition | FaceNet |
| Image Processing | OpenCV |
| Machine Learning | Scikit-learn |
| Backend | Flask |
| Frontend | HTML, CSS, JavaScript |
| Data Analysis | Pandas, Matplotlib |

## Model Evaluation

The project compares pretrained FaceNet with a partially fine-tuned FaceNet model.

| Evaluation Metric | Pretrained FaceNet | Fine-tuned FaceNet |
|---|---:|---:|
| ROC-AUC | 0.9808 | 0.9617 |
| Average Precision | 0.9582 | 0.9010 |

**Observation:** The pretrained FaceNet model achieved better verification performance than the partially fine-tuned model in this evaluation.

## Project Structure

```text
Face-Recognition-system/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── baseline_evaluation.py
│   ├── compare_verification_models.py
│   ├── finetune_facenet.py
│   ├── train_identity_classifier.py
│   └── evaluate_identity_classifier.py
│
├── results/
│   ├── baseline_metrics.csv
│   ├── pre_post_comparison.csv
│   └── evaluation visualizations
│
├── report/
│   └── figures/
│
├── static/
│   ├── style.css
│   └── results/
│
└── templates/
    └── index.html
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Amritha212/Face-Recognition-system.git
```

### 2. Open the project folder

```bash
cd Face-Recognition-system
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment (Windows)

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

## Dataset and Model Files

The original face dataset and trained model checkpoint files are not included in this public repository.

To run the complete training and evaluation workflows, provide an authorized dataset and the required model files separately.

## Project Highlights

- Implemented a face recognition and verification workflow.
- Evaluated pretrained and partially fine-tuned FaceNet models.
- Performed dataset quality checks and duplicate analysis.
- Developed a Flask-based evaluation dashboard.
- Analyzed model performance using verification metrics and visualizations.

## Author

**Amritha P**

GitHub: [Amritha212](https://github.com/Amritha212)

---

*This project was developed for learning and practical exploration of deep learning-based face recognition and verification.*