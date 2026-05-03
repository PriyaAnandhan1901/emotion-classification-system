# Emotion Classification System

A **7-class facial emotion recognition** system built using a custom CNN architecture trained on the **FER-2013 dataset**, with real-time webcam inference.

---

## Emotions Detected

| Label | Emotion |
|---|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Neutral |
| 5 | Sad |
| 6 | Surprise |

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **72%** |
| Dataset | FER-2013 |
| Training Images | ~28,000 |
| Classes | 7 emotions |
| Regularization | Dropout + Data Augmentation |

---

## Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** — CNN model training
- **OpenCV** — Face detection + webcam inference
- **NumPy** — Data processing
- **scikit-learn** — Evaluation metrics

---

## Project Structure

```
emotion-classification-system/
│
├── emotion_classification.py   # Main script — train or run webcam
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/PriyaAnandhan1901/emotion-classification-system.git
cd emotion-classification-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download FER-2013 Dataset
- Download from [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Extract and place as:
```
data/
└── FER-2013/
    ├── train/
    │   ├── angry/
    │   ├── happy/
    │   └── ...
    └── test/
        ├── angry/
        ├── happy/
        └── ...
```

### 4. Train the model
```bash
python emotion_classification.py train
```

### 5. Run real-time webcam detection
```bash
python emotion_classification.py
```

Press **`q`** to quit.

---

## How It Works

1. **Face Detection** — Haar cascade classifier detects faces in each webcam frame
2. **Preprocessing** — Face ROI is resized to 48x48 grayscale and normalized
3. **CNN Inference** — Custom 3-block CNN predicts emotion probabilities
4. **Overlay** — Emotion label + confidence score displayed on frame in real time

---

## Model Architecture

- 3 Convolutional blocks (Conv2D + BatchNorm + MaxPool + Dropout)
- 2 Fully Connected layers (512 → 256 → 7)
- Dropout regularization (0.25 and 0.5) to reduce overfitting
- Data augmentation: rotation, shift, zoom, horizontal flip

---

## Author

**Priya A**
- GitHub: [github.com/PriyaAnandhan1901](https://github.com/PriyaAnandhan1901)
- LinkedIn: [linkedin.com/in/priyaanandhan1901](https://linkedin.com/in/priyaanandhan1901)
- Email: priyaanandan682@gmail.com
