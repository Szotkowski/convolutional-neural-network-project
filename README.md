# Object Detection: Sea Shells vs. Lightbulbs
**Final Project for Computer Vision**

## 📌 Project Overview
This project implements a high-precision object detection system using the **RF-DETR (Transformer-based)** architecture. The system is designed to distinguish between organic shapes (**Sea Shells**) and manufactured objects (**Lightbulbs**) across diverse backgrounds.

The development followed a rigorous two-phase approach: a **Mini-Model** feasibility study followed by a **Full-Scale Model** (2,000 images) trained on high-performance computing clusters.

## 📊 Performance Summary
Based on the final evaluation logs from the production model:

| Category | mAP@50 | mAP@50:95 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Lightbulb** | 94.3% | 91.8% | 92.9% | 83.0% |
| **Sea Shell** | 84.4% | 81.7% | 84.7% | 83.0% |
| **Combined** | **89.4%** | **86.7%** | **88.8%** | **83.0%** |

---

## 📂 Project Structure
The repository is organized according to the project pipeline:

* **`01_zdrojova_videa/`**: Raw video data used for image extraction.
* **`02_dataset_coco/`**: The final 2,000-image dataset in COCO format (80/10/10 split).
* **`03_trenovany_model/`**: Final weights (`checkpoint_best_total.pth`) and training metrics.
* **`04_zaverecna_zprava/`**: Technical report and visual evaluation results.
* **`05_zdrojovy_kod_pro_trening/`**: Jupyter notebooks and Python scripts used for model training.
* **`mini_model/`**: Initial proof-of-concept training logs and weights.
* **`autoanotace_z_mini_model/`**: Results of the AI-assisted labeling phase.
* **`scripts/`**: Utility scripts for image selection (sharpness filtering) and dataset splitting.

---

## 🛠️ Technical Pipeline

### 1. Data Selection (Sharpness Filtering)
From a pool of 12,000 frames, we selected the top 2,000 images using **Laplacian Variance** to ensure high-quality, blur-free training data.

### 2. Model-in-the-Loop Annotation
We utilized the **Mini-Model** to perform auto-annotation on the larger dataset. By setting a low confidence threshold (0.05), we captured all potential objects, which were then manually verified in CVAT to ensure 100% ground-truth accuracy.

### 3. High-Performance Training
* **Architecture:** RF-DETR (Medium) with DINOv2 backbone.
* **Infrastructure:** MetaCentrum HPC (2x GPU).
* **Optimization:** EMA (Exponential Moving Average) weights for improved generalization.

---

## 🧠 Critical Evaluation
* **Mini-Model Significance:** The initial 200-image test was crucial for identifying a category ID mapping error. Resolving this early prevented wasted compute cycles during the big-model phase.
* **Class Performance:** Lightbulbs achieved higher precision (94.3% mAP@50) due to consistent industrial geometry. Sea shells (84.4% mAP@50) showed higher variance but were successfully modeled.
* **Robustness:** The high combined recall (83%) indicates the model is highly reliable for real-world automated sorting tasks.

---

## 🚀 Usage
To run inference using the final model:

```python
from rfdetr import RFDETRMedium

model = RFDETRMedium.from_checkpoint('03_trenovany_model/checkpoint_best_total.pth')
results = model.predict('input_image.jpg', conf_threshold=0.5)
results.show()