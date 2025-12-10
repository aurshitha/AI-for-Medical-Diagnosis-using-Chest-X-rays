<p align="center">
  <img src="https://img.shields.io/badge/Chest%20Xray%20Classification-DenseNet121-0059ff?style=for-the-badge&logo=github&logoColor=white" />
</p>


# Chest X-ray Multi-Label Pathology Classification (DenseNet-121)

This project implements a robust deep learning pipeline for multi-label classification of thoracic pathologies from chest X-ray images. It utilizes Transfer Learning with the DenseNet-121 architecture, adapting it to diagnose 14 common chest conditions simultaneously, such as Pneumonia, Effusion, and Cardiomegaly.

The core focus of this project is not only high performance (measured via AUC) but also interpretability through Grad-CAM (Gradient-weighted Class Activation Mapping), allowing for visual confirmation of the model's diagnostic reasoning.

## 🚀 Key Features

### **• Multi-Label Classification**
Predicts the presence or absence of **14 thoracic pathologies simultaneously**, enabling efficient and scalable diagnostic support.

### **• Two-Phase Transfer Learning**
Fine-tunes a pre-trained **DenseNet-121** model using a structured two-step approach:
1. **Train only the custom classification head**  
2. **Unfreeze and fine-tune base DenseNet layers** with a very low learning rate  
This approach improves model stability and convergence for medical imaging tasks.

### **• Robust Class Weighting**
Applies **Inverse Frequency Weighting** to handle the severe **class imbalance** typical in medical datasets, ensuring the model learns rare pathologies effectively.

### **• AUC-Optimized Training**
Trains and evaluates the model using **AUC (Area Under the ROC Curve)** — a reliable metric for highly imbalanced multi-label medical classification problems.

### **• Interpretability with Grad-CAM**
Generates **visual heatmaps** highlighting the most influential regions of the X-ray image, enabling clinicians to **validate model reasoning** and improve model trustworthiness.


## 🛠️ Setup and Prerequisites

This project is optimized for execution within Google Colab, leveraging GPU acceleration and interactive file upload features.

## 1. Environment Setup

```bash
  # Required libraries
  pip install tensorflow keras numpy pandas scikit-learn matplotlib
```
## 2. Data
This pipeline assumes the use of the NIH Chest X-ray Dataset (or a sample thereof).
Place your data files according to the following structure:
| File/Folder     | Path Example                                                | Description                                                      |
|-----------------|--------------------------------------------------------------|------------------------------------------------------------------|
| Images          | /content/sample_chest_xray_data/sample/images/               | All X-ray images (.png files).                                  |
| CSV Labels      | /content/sample_chest_xray_data/sample_labels.csv            | Metadata file containing the Image Index and Finding Labels.     |
| Model Save      | /content/drive/MyDrive/ChestXRay_Models/                     | Persistent storage location (Google Drive) to save the model.    |

## 📦 Project Workflow

The project workflow is divided into two main phases: **Training & Saving** and **Loading & Interpreting** the model.

---

##  Phase I: Model Training and Saving

This phase includes data loading, cleaning, applying robust class weighting, performing two-phase transfer learning, and finally saving the optimized model to Google Drive.

### **A. Training Pipeline (`train_model.py` equivalent)**

This step covers the core training logic such as:

- Dataset loading and preprocessing  
- Handling class imbalance using **Inverse Frequency Class Weights**  
- Two-phase training:
  1. Training the custom classification head  
  2. Fine-tuning the DenseNet-121 base layers with a low learning rate  

> **Note:**  
> The complete training code is omitted for brevity since it was run inside the Colab notebook, but the full workflow is described above.

---

### **B. Saving the Model**

After training, the best-performing model (based on **Validation AUC**) is saved permanently to Google Drive for future use.

```python
import tensorflow as tf
from google.colab import drive
import os

# 1. Mount Google Drive
drive.mount('/content/drive')

# 2. Define Paths (These must match your setup)
DRIVE_DIR = '/content/drive/MyDrive/ChestXRay_Models'
FINAL_MODEL_PATH = os.path.join(DRIVE_DIR, 'optimized_densenet_chest_xray.keras')
TEMP_MODEL_PATH = "/content/nih_chest_xray_optimized_model.keras" # Saved by ModelCheckpoint

# Ensure the Drive directory exists
os.makedirs(DRIVE_DIR, exist_ok=True)

# 3. Load the best temporary model and save it permanently
try:
    # Load the temporary best model with custom AUC metric
    LABELS = ['Atelectasis', ..., 'Pneumothorax'] # Define all 14 labels
    loaded_model = tf.keras.models.load_model(
        TEMP_MODEL_PATH, 
        custom_objects={"auc": tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(LABELS))}
    )
    
    # Save to Drive
    tf.keras.models.save_model(loaded_model, FINAL_MODEL_PATH)
    print(f"✅ Model successfully saved to Google Drive at: {FINAL_MODEL_PATH}")

except Exception as e:
    print(f"Error during model saving: {e}")
```

##  Phase II: Prediction and Interpretability (Grad-CAM)
This script is used in any future session to load the saved model, upload a new X-ray, predict the disease, and visualize the model's reasoning.

###  **A. Full Prediction and Grad-CAM Script**
This comprehensive block handles all steps: loading the persistent model, uploading the test image, prediction, and visual explanation.
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from google.colab import drive, files
import os

# --- 1. CONFIGURATION ---
TARGET_SIZE = (224, 224)
LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
DRIVE_DIR = '/content/drive/MyDrive/ChestXRay_Models'
FINAL_MODEL_PATH = os.path.join(DRIVE_DIR, 'optimized_densenet_chest_xray.keras')
LAST_CONV_LAYER_NAME = 'conv5_block16_concat' # Last conv layer of DenseNet-121

# --- 2. HELPER FUNCTIONS (Prediction and Grad-CAM) ---

def prepare_image_for_prediction(img_path):
    """Loads, resizes, and preprocesses a single image."""
    img = image.load_img(img_path, target_size=TARGET_SIZE, color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    """Computes the Grad-CAM heatmap."""
    # Create a model to output both feature maps and predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def superimpose_heatmap(img_path, heatmap):
    """Superimposes the heatmap onto the original image."""
    img = image.load_img(img_path)
    img = image.img_to_array(img)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def predict_and_run_gradcam(model, img_path):
    """Runs prediction, prints results, and generates Grad-CAM for the top prediction."""
    if model is None: return

    # --- Prediction ---
    processed_image = prepare_image_for_prediction(img_path)
    prediction = model.predict(processed_image, verbose=0)[0] 
    
    results = pd.DataFrame({
        'Disease': LABELS,
        'Confidence (%)': prediction * 100
    }).sort_values(by='Confidence (%)', ascending=False).reset_index(drop=True)
    
    top_disease = results.iloc[0]
    print("\n--- Model Prediction Results ---")
    print(f"Highest Confidence Prediction: {top_disease['Disease']} ({top_disease['Confidence (%)']:.2f}%)")
    print("\nAll Predictions:\n", results.to_markdown(index=False, floatfmt=".2f"))

    # --- Grad-CAM Execution (on the top predicted disease) ---
    TARGET_DISEASE = top_disease['Disease']
    TARGET_CONFIDENCE = top_disease['Confidence (%)']
    target_index = LABELS.index(TARGET_DISEASE)
    
    print(f"\n--- Generating Grad-CAM for: {TARGET_DISEASE} ---")
    
    try:
        heatmap = make_gradcam_heatmap(processed_image, model, LAST_CONV_LAYER_NAME, pred_index=target_index)
        gradcam_image = superimpose_heatmap(img_path, heatmap)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(image.load_img(img_path)); axes[0].set_title("Original X-Ray"); axes[0].axis("off")
        axes[1].imshow(gradcam_image); axes[1].set_title(f"Grad-CAM for: {TARGET_DISEASE} ({TARGET_CONFIDENCE:.2f}%)"); axes[1].axis("off")
        plt.show() # 
    except Exception as e:
        print(f"An error occurred during Grad-CAM generation: {e}")


# --- 3. EXECUTION BLOCK ---
print("1. Mounting Google Drive...")
drive.mount('/content/drive', force_remount=True)

loaded_model = None
print("\n2. Loading Model from Drive...")
try:
    loaded_model = tf.keras.models.load_model(FINAL_MODEL_PATH, custom_objects={"auc": tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=len(LABELS))})
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Interactive Upload and Prediction
if loaded_model:
    print("\n3. Please select the X-ray image file you wish to test.")
    try:
        uploaded = files.upload()
        if uploaded:
            uploaded_file_name = next(iter(uploaded))
            NEW_IMAGE_PATH = f"/content/{uploaded_file_name}"
            print(f"\nFile uploaded successfully. Starting analysis on: {uploaded_file_name}")
            predict_and_run_gradcam(loaded_model, NEW_IMAGE_PATH)
        else:
            print("No file was uploaded. Analysis aborted.")
    except Exception as e:
        print(f"An error occurred during file upload or analysis: {e}")
```

## 📊 Model Performance Summary
The model's final performance is measured by the Macro-Average Area Under the Curve (AUC) on the independent test set.

| Metric                           | Value   | Interpretation                                                              |
|----------------------------------|---------|-------------------------------------------------------------------------------|
| Overall Macro-Average AUC        | 0.6344  | Significant improvement over random chance (0.50).                           |
| Best Class AUC (Emphysema)       | 0.7850  | High diagnostic power for this pathology.                                    |
| Worst Class AUC (Fibrosis)       | 0.4408  | Indicates difficulty in classifying this rare disease due to limited samples. |
