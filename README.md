# AI-for-Medical-Diagnosis-using-Chest-X-rays

A deep learning-based system for automated classification of thoracic pathologies from chest X-ray images, featuring explainable AI for clinical transparency.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Academic-blue)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)

---

## 🎯 Overview

This project develops an AI-powered diagnostic model that analyzes chest X-ray images to classify medical conditions using deep learning. Transfer learning with DenseNet121 is used to improve performance on a limited dataset. The system also incorporates Grad-CAM to generate visual explanations, providing transparency behind model predictions.

---

## ✨ Features

- **Image Preprocessing**
  - Image resizing and normalization  
  - Train–test split  
  - Data augmentation to improve model generalization  

- **Deep Learning Model**
  - Transfer learning using **DenseNet121**
  - Fine-tuned layers for classification  
  - Trained using TensorFlow/Keras  

- **Performance Evaluation**
  - Accuracy and loss analysis  
  - Training vs. validation metrics  

- **Explainable AI**
  - **Grad-CAM heatmaps** to highlight important diagnostic regions  

---

## 🚀 Installation

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/aurshitha/ai-for-medical-diagnosis-using-chest-x-rays.git
cd ai-for-medical-diagnosis-using-chest-x-rays

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Dataset

The project uses a subset of a chest X-ray dataset for training and evaluation.

### Dataset Characteristics

- Preprocessed to **224×224** resolution  
- Stored in `.png` / `.jpg` format  
- Labels multi-class / multi-disease  
- Split into:
  - 70% Training  
  - 15% Validation  
  - 15% Test  

---

## 🚀 Basic Usage

```python
# Load trained model
model = load_model('models/densenet121_chestxray.h5')

# Preprocess input image
image = preprocess_xray('path_to_xray.jpg')

# Generate prediction
preds = model.predict(image)

# Generate Grad-CAM explanation
heatmap = generate_gradcam(model, image, layer_name='conv_layer')
```

---

## 🧰 Technologies Used

- Python 3.8+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- OpenCV / PIL

---

## 🧠 Learnings

This project helped develop practical skills in:

- Medical image preprocessing  
- Transfer learning for limited data  
- Training and tuning deep learning models  
- Evaluating model performance  
- Applying Grad-CAM for visual interpretability  
- Understanding challenges of AI in healthcare

---
