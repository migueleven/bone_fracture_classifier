#  **Fracture Detection in X-ray Images**  

> A machine learning project to classify fractures in X-ray images using transfer learning and advanced fine-tuning techniques.

---

## 📖 **Table of Contents**  
- [📁 Project Overview](#-project-overview)  
- [📊 Dataset](#-dataset)  
- [⚙️ Workflow](#%EF%B8%8F-workflow)  
- [💻 Tools and Technologies](#-tools-and-technologies)  
- [🧠 Model Architecture](#-model-architecture)  
- [📈 Results](#-results)  
- [🚀 Future Improvements](#-future-improvements)  

---

## 📁 **Project Overview**  
This project aims to detect fractures in X-ray images by leveraging a pre-trained ResNet50 model through transfer learning and fine-tuning. The goal is to assist medical professionals in identifying fractures efficiently and accurately.  

- **Problem**: Detect fractured vs. non-fractured bones.  
- **Solution**: Develop a binary classification model using advanced machine learning techniques.  
- **Objective**: Achieve high accuracy and robust generalization on unseen X-ray images.  

---

## 📊 **Dataset**  
- **Source**: Custom X-ray dataset with two categories: `fractured` and `non-fractured`.  
- **Size**:  
  - Training: ~4000 images.  
  - Validation: ~1000 images.  
  - Test: ~1000 images.  
- **Preprocessing**:  
  - Grayscale conversion to 3 channels for compatibility with pre-trained models.  
  - Normalization to match ImageNet statistics.  
  - Augmentations: Random cropping, flipping, resizing.  

---

## ⚙️ **Workflow**  

1. **Data Preparation**:  
   - Normalized, resized, and augmented the dataset.  
   - Ensured balanced class distribution.  

2. **Exploratory Data Analysis (EDA)**:  
   - Visualized dataset distribution and augmented samples.  
   - Checked for class imbalance.  

3. **Model Training**:  
   - Used ResNet50 pre-trained on ImageNet.  
   - Fine-tuned specific layers while freezing others.  
   - Applied Early Stopping and learning rate scheduling.  

4. **Evaluation**:  
   - Tested on unseen data.  
   - Computed metrics: Accuracy, F1-score, Precision, Recall.  
   - Visualized confusion matrices and performance curves.  

---

## 💻 **Tools and Technologies**  

- **Programming Language**: Python 🐍  
- **Libraries**:  
  - PyTorch (Model building and training)  
  - torchvision (Transfer learning, preprocessing)  
  - scikit-learn (Metrics and evaluation)  
  - matplotlib, seaborn (Visualizations)  

- **Hardware**:  
  - GPU-enabled system (NVIDIA CUDA) for efficient training.  

---

## 🧠 **Model Architecture**  

- **Base Model**: ResNet50  
  - Pre-trained on ImageNet.  
  - Final fully connected layer modified to output 2 classes.  
- **Optimizer**: Adam  
- **Loss Function**: CrossEntropyLoss  
- **Learning Rate Scheduler**: StepLR (decays every 7 epochs).  

---

## 📈 **Results**  

| Metric       | Original Model (Frozen)  | Fine-Tuned Model (Unfrozen) |  
|--------------|--------------------------|-----------------------------|  
| Accuracy     | **0.80**                 | **0.99**                    |  
| F1-Score     | **0.80**                 | **0.99**                    |  
| Precision    | **0.80**                 | **0.985**                   |  
| Recall       | **0.80**                 | **0.985**                   |  

### **Confusion Matrix (Fine-Tuned Model)**  
 
![Captura de pantalla 2025-01-17 160816](https://github.com/user-attachments/assets/4b900ef3-1ed7-4aff-bf8c-8bf1ccc4538a)


---

## 🚀 **Future Improvements**  

1. **Augmented Dataset**: Include more samples to improve generalization.  

---

## 📂 **Project Structure**  

```plaintext
📦 Fracture-Detection-Project  
├── 📁 data/                # Contains train/val/test datasets  
├── 📁 notebooks/           # Jupyter notebooks for training and EDA  
├── 📁 models/              # Saved model weights (e.g., model.pth)  
├── 📁 results/             # Evaluation metrics and visualizations  
└── README.md               # Project README  
