# 🏆 Fingerprint Spoofing Detection – Machine Learning & Pattern Recognition

📌 **Course**: Machine Learning and Pattern Recognition  
📌 **Institution**: Politecnico di Torino – A.A. 2023/2024  
📌 **Project Objective**: Evaluation and comparison of different models and pre-processing/post-processing techniques for fingerprint spoofing detection.

---

## 🚀 Introduction  

This project focuses on solving a **binary classification problem**, aiming to detect **fingerprint spoofing**. Specifically, the task involves distinguishing between **genuine** and **fake** fingerprint images.  

The dataset provided contains labeled examples, with the labels indicating whether a fingerprint is genuine (True, label 1) or counterfeit (False, label 0). Each sample is represented using six features, which have been extracted to capture the high-level characteristics of the fingerprint images.

---

## 📁 Repo Structure  

The dataset was analyzed through various steps, beginning with **data visualization** and **dimensionality reduction techniques** (PCA and LDA). The evaluation of different models started in laboratory 5, and the following models were compared:

- Gaussian Models
- Logistic Regression
- Support Vector Machines (SVM)
- Gaussian Mixture Models (GMM)

### 📊 Performance Evaluation  

The performance of each model was evaluated using the following metrics:
- **Accuracy**
- **DCF (Detection Cost Function)**
- **minDCF (Minimum Detection Cost Function)**

All models were trained, validated, and tested using a **K-fold cross-validation approach**.

### 📑 Detailed Results  
All the results, including detailed plots and comparisons, are summarized in the [Report](Report.pdf).

---

