# AI-for-Medical-Diagnosis-using-Chest-X-rays

A deep learning-based system for automated multi-label classification of thoracic pathologies from chest X-ray images, featuring explainable AI for clinical transparency.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-Academic-blue)
![Status](https://img.shields.io/badge/Status-Research%20Project-green)

## 🎯 Overview

This project develops an AI-powered diagnostic tool that identifies **14 different thoracic pathologies** from chest X-ray images using convolutional neural networks (CNNs) and transfer learning. The system incorporates explainable AI (XAI) techniques with Grad-CAM visualizations to provide transparent predictions for clinical use.

## ✨ Features

- **🔬 Multi-label Classification**: Detects 14 thoracic diseases simultaneously
- **🔄 Transfer Learning**: Utilizes pre-trained models (DenseNet121, ResNet, VGG16)
- **⚖️ Class Imbalance Handling**: Weighted loss functions and advanced data augmentation
- **👁️ Explainable AI**: Grad-CAM heatmaps for prediction interpretability
- **🏥 Clinical Integration**: Patient metadata (age, gender, view position) support
- **☁️ Cloud Ready**: Developed on Google Colab for easy replication

## 🚀 Installation

### Prerequisites
```bash
# Clone the repository
git clone https://github.com/aurshitha/ai-for-medical-diagnosis-using-chest-x-rays.git
cd chest-xray-ai-diagnosis

# Install dependencies
pip install -r requirements.txt

## 📁 Dataset

The project uses the **NIH ChestX-ray14 dataset** subset containing:

### 📊 Dataset Statistics
- **🖼️ 5,606 chest X-ray images**
- **🏷️ 14 pathological conditions + "No Finding"**
- **📋 Patient metadata**: age, gender, view position
- **🎯 Multi-label classification** support
- **⚖️ Class imbalance** addressed through weighted loss

### 🏥 Pathology Labels

| Condition | Description | Frequency |
|-----------|-------------|-----------|
| **Atelectasis** | Collapse or closure of lung | 🔵 |
| **Cardiomegaly** | Enlarged heart | 🔵 |
| **Consolidation** | Lung tissue filled with liquid | 🔵 |
| **Edema** | Fluid accumulation in lungs | 🔵 |
| **Effusion** | Excess fluid around lungs | 🔵 |
| **Emphysema** | Damage to lung air sacs | 🟡 |
| **Fibrosis** | Lung tissue scarring | 🟡 |
| **Hernia** | Diaphragm hernia | 🔴 |
| **Infiltration** | Abnormal substance in lungs | 🔵 |
| **Mass** | Abnormal growth in lungs | 🔵 |
| **Nodule** | Small lung abnormality | 🔵 |
| **Pleural Thickening** | Pleura membrane thickening | 🔵 |
| **Pneumonia** | Lung inflammation | 🔵 |
| **Pneumothorax** | Collapsed lung | 🔵 |
| **No Finding** | Normal chest X-ray | 🟢 |

**Legend**: 
- 🟢 Common (>1000 cases)
- 🔵 Moderate (100-1000 cases) 
- 🟡 Rare (50-100 cases)
- 🔴 Very Rare (<50 cases)

### 📝 Dataset Characteristics
- **Image Resolution**: 1024×1024 pixels (original)
- **Preprocessed Size**: 224×224 or 320×320 pixels
- **Format**: Grayscale/DICOM converted to PNG/JPG
- **Labels**: Multi-hot encoded vectors
- **Split**: 70% Train, 15% Validation, 15% Test

### 🔗 Dataset Access
```python
# Download via Kaggle API
!kaggle datasets download -d nih-chest-xrays/sample
!unzip sample.zip -d data/raw/

## 🚀 Basic Usage

```python
# Load trained model
model = load_model('models/densenet121_chestxray.h5')

# Preprocess image
image = preprocess_xray('path_to_image.jpg')

# Make prediction
predictions = model.predict(image)

# Generate explanation
heatmap = generate_gradcam(model, image, layer_name='final_conv_layer')

# 🧰 Technologies

- Programming Language: Python 3.8+
- Deep Learning Framework: TensorFlow/Keras
- Development Environment: Google Colab
- Image Processing: OpenCV, PIL
- Data Analysis: Pandas, NumPy, Matplotlib, Seaborn

# Models Used:
- DenseNet121 (Primary): 94-95% accuracy
- Custom CNN: 91.05% accuracy
- ResNet18: 86.52% accuracy
- AlexNet: 85.93% accuracy
