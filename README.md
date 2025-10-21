# Helmet Violation Detection from Indian CCTV Video

## Problem Statement
Detect and flag two-wheeler helmet violations (helmetless riding) from traffic camera frames in Indian cities in real-time.

## Description
This project creates a computer vision system using YOLOv8 and object tracking to detect two-wheeler riders and classify helmet usage. Optionally, license plate OCR is performed for enforcement.

## Dataset
- **Indian Helmet Detection Dataset**
- Research-generated dataset of Indian two-wheeler violations (images and video with annotations for helmet & plate)

## Project Structure

```
.
├── data/
│   ├── external/
│   ├── processed/
│   └── raw/
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       ├── test/
│       └── valid/
│           ├── images/
│           └── labels/
├── models/
├── notebooks/
│   └── HELMET_VIOLATION_DETECTION_v4.0.ipynb
├── results/
│   └── logs/
├── scripts/
├── requirements.txt
├── environment.yml
└── README.md
```

## Week-by-Week Plan (8 Weeks)

- **Week 1:** Review datasets and annotation format; frame extraction from video
- **Week 2:** Preprocess and augment images; split train/validation/test
- **Week 3:** Train YOLOv8 for bike-rider detection; evaluate mAP
- **Week 4:** Train helmet/no-helmet classifier module; fine-tune thresholds
- **Week 5:** Implement license plate OCR (easyOCR or Tesseract) for flagged riders
- **Week 6:** Integrate into real-time pipeline: detection → classification → OCR
- **Week 7:** Test on hold-out video streams; evaluate precision/recall and false positives
- **Week 8:** Prepare final demo system, performance report, enforcement workflow documentation

## Submission Requirements

- Trained model weights
- Inference pipeline (code + notebook or demo app)
- Confusion matrix and violation simulations
- License plate OCR results on violations
- Final report describing commercial use for municipal enforcement and safety impact

## Notebooks

- Main EDA, augmentation, and pipeline code is in [`notebooks/HELMET_VIOLATION_DETECTION_v4.0.ipynb`](notebooks/HELMET_VIOLATION_DETECTION_v4.0.ipynb)

## Key Techniques Used

- YOLOv8 object detection for rider, helmet, and motorbike
- Data augmentation: flipping, zooming, mosaic, cutout, synthetic weather, grayscale, noise injection, rotation, shadow, edge detection
- Exploratory data analysis: class distribution, bounding box analysis, co-occurrence, brightness, background bias
- License plate OCR (planned: easyOCR/Tesseract)
- Real-time inference pipeline

## How to Run

1. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```
2. Run the main notebook for EDA, augmentation, and training:
    - Open [`notebooks/HELMET_VIOLATION_DETECTION_v4.0.ipynb`](notebooks/HELMET_VIOLATION_DETECTION_v4.0.ipynb) in Jupyter or VS Code.
3. Follow week-by-week plan for full pipeline development.

---

For questions or contributions, please open an issue or pull request.