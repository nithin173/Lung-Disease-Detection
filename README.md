# 🫁 Lung Disease Detection from Chest X-rays using Vision Transformers (ViT) with 3D Reconstruction

This project focuses on detecting lung diseases (such as pneumonia,tuberculosis, etc.) using chest X-ray images. It uses **Vision Transformers (ViT)** for classification and incorporates **3D image reconstruction** to enhance diagnostic accuracy.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup Instructions](#setup-instructions)


---

##  Project Overview

This project aims to:
- Leverage **Vision Transformers** for high-performance medical image classification.
- Use spatial correlations in 2D chest X-rays to reconstruct a **3D volumetric view** of the lungs.
- Improve diagnosis by capturing subtle features not easily seen in 2D images alone.

---

## 📂 Dataset

We used a publicly available chest X-ray dataset. The dataset is **not included** in this repository due to size limitations.

📥 Download the dataset from: https://www.kaggle.com/datasets/omkarmanohardalvi/lungs-disease-dataset-4-types

After downloading, place the extracted contents in the `lungs-disease-dataset-4-types` directory.

---

##  Model Architecture

- ✅ **Vision Transformer (ViT)** for 2D X-ray classification
- ✅ Evaluation Metrics: Accuracy, Confusion Matrix, Precision, Recall

Technologies Used:
- PyTorch & Torchvision
- Transformers (Hugging Face)
- NumPy, Pandas, OpenCV
- Matplotlib & Seaborn for visualization

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/Lung-Disease-Detection.git
cd Lung-Disease-Detection
```
- Now Download the trained model using link: 
- https://drive.google.com/file/d/1A-PkuZ0ATG7Hu9PchY_kwcCljc3DByaM/view?usp=sharing
- Then ope app.py file  in any development environment
- Run the file 
- The open the link which was visible in terminal
- Then upload any X-Ray Image it will classify which disease it is.. 
