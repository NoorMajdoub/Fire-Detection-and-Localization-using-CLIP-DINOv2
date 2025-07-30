
#  Fire Detection and Localization Using CLIP and DINOv2

##  Project Overview

This Jupyter notebook implements a two-stage pipeline for **automatic fire detection and localization** in forest imagery using **state-of-the-art vision-language and self-supervised models**:

1. **Fire Detection** – Binary classification using OpenAI’s CLIP-ViT model.
2. **Fire Localization** – Patch-level anomaly detection using Facebook’s DINOv2 model with cosine similarity for feature matching.

---

##  Methodology

###  1. Fire Detection with CLIP-ViT

* Utilizes [CLIP](https://openai.com/research/clip) (Contrastive Language–Image Pretraining) for zero-shot classification.
* Embeds image and prompt texts into a shared semantic space.
* Prompts used:

  * `"a normal forest scene"`
  * `"a fire or smoke in a forest"`
* Classification is based on cosine similarity between image and text embeddings.
* Outputs class label (`fire` / `no fire`) along with probability scores.

### 🗺️ 2. Fire Localization with DINOv2

* Uses [DINOv2](https://github.com/facebookresearch/dinov2) for self-supervised feature extraction.
* Extracts patch-level features from both:

  * Target image
  * A reference *normal* forest image (used as baseline context)
* Computes cosine similarity between patches to localize anomalous (i.e. fire-related) regions.
* Visualizes results with anomaly heatmaps overlaid on the original image.

---

## ⚙️ Setup & Dependencies

Install the following libraries before running the notebook:

```bash
pip install torch torchvision transformers opencv-python matplotlib pillow
```

**Required Libraries:**

* Python 3.8+
* PyTorch (CUDA support recommended)
*  Transformers (`CLIP`, `DINOv2`)
* OpenCV, PIL, NumPy, Matplotlib

---

## 🚀 Usage

### 🔥 Fire Detection

```python
result, probs = detectfile("/path/to/image.jpg")
print(f"Fire Detected: {bool(result)} | Probabilities: {probs}")
```

### 🔍 Fire Localization

* Load DINOv2 model and a reference image (non-fire scene).
* Extract patch features from both test and reference images.
* Compute per-patch cosine similarity.
* Generate and display heatmap highlighting anomalous regions.

---


##  Notes

* Developed and tested in a Kaggle notebook environment.
* Models are sourced from Hugging Face Hub.
* Includes fallback logic for CUDA initialization issues.
* Detection thresholding is static; performance can be optimized via calibration.
* Localization assumes a clean reference image (`AoF06726.jpg`) for anomaly comparison.

---

##  Future Work

* Train or fine-tune models for improved robustness and generalization.
* Extend pipeline to support:

  * **Video input**
  * **Real-time inference**
* Replace static thresholding with adaptive scoring (e.g., via Gaussian models or learned thresholds).
* Automate reference image selection or compute scene-level context clusters.

---



This project is provided **as-is** for **research and educational purposes**.
Please review the licenses of:



