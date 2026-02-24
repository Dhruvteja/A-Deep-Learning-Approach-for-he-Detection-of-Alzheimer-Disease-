# 🧠 A Deep Learning Approach for the Detection of Alzheimer Disease

## 📘 Project Overview

This repository presents a deep learning-based approach for detecting **Alzheimer’s Disease (AD)** using MRI brain images. The project was developed as part of the Bachelor of Engineering (B.E.) degree in Computer Science and Engineering (2022–2023).

> 📄 Full Project Documentation: Alzheimer_s Documentation.pdf

The system leverages the **DenseNet-169 architecture** and transfer learning techniques to classify MRI images into different stages of Alzheimer’s Disease.

---

## 🎓 Academic Information

**Degree:** Bachelor of Engineering (B.E.)
**Branch:** Computer Science and Engineering
**Institution:** Methodist College of Engineering and Technology
**Affiliated to:** Osmania University
**Academic Year:** 2022–2023

### 👨‍💻 Team Members

* Santosh Naga Manideep Bhonagiri (160719733002)
* Tauseef Ali Khan (160719733003)
* Dhruv Teja Manjrekar (160719733023)

**Project Guide:**
Mr. A.A.R. Senthil Kumar, Assistant Professor, Dept. of CSE

---

## 📌 Abstract

Alzheimer’s Disease is a progressive neurodegenerative disorder characterized by memory loss and cognitive decline. Early diagnosis is essential for timely intervention and management.

This project proposes a **Deep Learning approach using DenseNet-169** for robust and accurate prediction of Alzheimer’s stages from MRI images. The dataset consists of four classes:

* Mild Demented
* Moderate Demented
* Non-Demented
* Very Mild Demented

The model uses a sequential architecture with additional layers such as:

* Dense layers
* Flatten layer
* Batch Normalization
* Dropout
* Softmax activation

Callback mechanisms like **Early Stopping** and **Model Checkpoints** are used to improve performance and prevent overfitting.

---

## 🧠 About Alzheimer’s Disease

Alzheimer’s Disease (AD) is:

* A common form of dementia
* Characterized by progressive cognitive decline
* Ranked among the leading causes of death worldwide

### 🔎 Major Symptoms

* Persistent memory loss
* Difficulty in reasoning and decision-making
* Language problems
* Mood and personality changes
* Difficulty performing daily tasks

### ⚠️ Risk Factors

* Age (Risk doubles after 65)
* Family history
* Down Syndrome
* Head injuries
* Cardiovascular conditions (diabetes, hypertension, obesity)
* Lifestyle factors (smoking, inactivity)

### ✅ Benefits of Early Detection

* Accurate diagnosis
* Better treatment planning
* Improved quality of life
* Participation in research trials
* Reduced stigma

---

## 🤖 Introduction to Deep Learning

Deep learning uses artificial neural networks inspired by the human brain to learn patterns from high-dimensional data.

It is widely used in:

* Image recognition
* Speech processing
* Natural language processing
* Medical diagnosis

### 🔬 DenseNet-169 Model

DenseNet-169:

* Contains 169 layers
* Solves vanishing gradient problem
* Uses dense connectivity between layers
* Requires fewer parameters compared to traditional CNNs

The CNN architecture consists of:

* Convolution Layer
* Pooling Layer
* Fully Connected Layer

---

## 🗂 Dataset Description

The dataset consists of MRI brain images classified into four categories:

1. Mild Demented
2. Moderate Demented
3. Non-Demented
4. Very Mild Demented

### 📊 MRI in Machine Learning

MRI:

* Is a non-invasive imaging technique
* Provides detailed brain structure images
* Does not use ionizing radiation
* Is ideal for neurological diagnosis

Machine learning applications of MRI include:

* Disease detection
* Image segmentation
* Treatment response prediction
* Brain-computer interfaces

---

## 🧩 Stages of Alzheimer’s Disease

1. **Preclinical (Very Mild)**
   Brain changes occur before symptoms appear.

2. **Early Stage (Mild)**
   Memory loss and mild confusion.

3. **Middle Stage (Moderate)**
   Increased cognitive impairment and behavioral changes.

4. **Severe Stage**
   Complete dependency and loss of communication abilities.

---

## 📚 Literature Survey Summary

A comprehensive survey of over 100 research papers was conducted. Key observations:

* Most studies focus on binary classification (AD vs HC).
* Common datasets: ADNI, OASIS, Kaggle MRI datasets.
* Algorithms used:

  * SVM
  * Random Forest
  * Decision Trees
  * CNN
  * DNN
  * Ensemble Learning
* Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1-score
  * AUC
  * Confusion Matrix

Our project focuses on **categorical multi-class classification**, addressing a gap in previous studies.

---

## 🏗 System Architecture

### Major Components

1. Data Collection
2. Data Pre-processing
3. Pre-trained Model (DenseNet-169)
4. Model Building
5. Feature Extraction
6. Model Training
7. Model Evaluation
8. Testing

### ⚙️ Pre-processing Steps

* Data loading
* Label encoding
* Train-validation split
* Data augmentation
* Shuffling

---

## 🧪 Model Implementation

### Model Building

* DenseNet-169 as base model
* Custom classification layers
* ReLU activation
* Dropout for regularization
* Softmax output layer

### Training Strategy

* Adam Optimizer
* Early Stopping
* Model Checkpoint
* Learning rate adjustments

---

## 📈 Evaluation & Results

### Performance Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* AUC
* Confusion Matrix

### Analysis Includes

* Training & Validation Loss Graph
* Accuracy Graph
* Train, Validation & Test Confusion Matrices
* Classification Reports
* Model Comparison

The model demonstrates strong classification capability for detecting Alzheimer’s stages from MRI images.

---

## 🖥 System Implementation & UI

The system includes:

* A user interface for MRI upload
* Stage prediction output
* Accuracy display
* Test case demonstrations for all four stages

---

## 🔮 Future Scope

* Improve dataset size and diversity
* Explore multimodal learning (MRI + PET)
* Improve feature extraction techniques
* Enhance model generalization
* Apply 3D CNN-based DenseNet architectures

---

## 🏁 Conclusion

Early detection of Alzheimer’s Disease is crucial for improving patient outcomes.

This project successfully implements a DenseNet-169 based deep learning model capable of:

* Multi-class stage classification
* Improved accuracy through transfer learning
* Effective MRI-based disease prediction

The system provides a promising foundation for future research in AI-driven neurological diagnostics.

---

## 📚 References

The project references multiple research works related to:

* Deep learning in medical imaging
* CNN-based Alzheimer detection
* MRI-based classification models
* Ensemble learning approaches

(Refer to the complete documentation for full bibliography.)

---

## 📄 License

This project is developed for academic and research purposes.

---

⭐ If you find this repository useful, consider giving it a star!
