🫁 Lung Cancer Prediction using Machine Learning & Deep Learning
📌 Project Overview

This project focuses on predicting whether a person has lung cancer based on clinical and lifestyle attributes using multiple machine learning and deep learning models. The goal is to compare different modeling approaches and identify the most effective method for lung cancer detection.

The project uses tabular medical data and evaluates traditional ML models against a neural network classifier.

🎯 Objectives

Predict the presence of lung cancer (Yes / No)

Compare performance of different classification models

Analyze model effectiveness using proper medical evaluation metrics

Provide model interpretability through feature importance

📊 Dataset Description

The dataset contains patient-related attributes such as:

Age

Gender

Smoking habits

Alcohol consumption

Chronic disease history

Respiratory symptoms (wheezing, coughing, chest pain, etc.)

Target Variable:

LUNG_CANCER

1 → Lung cancer present

0 → Lung cancer not present

🛠️ Models Implemented

The following models were trained and evaluated:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Neural Network (PyTorch – MLP)

⚙️ Data Preprocessing

Categorical variables encoded numerically

Rows with missing target values removed

All features converted to numeric format

Missing values handled using mean imputation

Feature scaling applied for neural network training

Stratified train-test split to handle class imbalance

📈 Evaluation Metrics

Models were evaluated using:

Accuracy

Precision

Recall (Sensitivity)

F1-score

ROC–AUC score

Confusion Matrix

ROC and Precision–Recall curves

Special emphasis was placed on Recall, as minimizing false negatives is critical in medical diagnosis.

🏆 Results Summary

XGBoost achieved the best overall performance in terms of recall and ROC–AUC.

Neural Network performed competitively but did not outperform XGBoost on tabular data.

Traditional models like Logistic Regression and Random Forest provided strong baselines.

🧠 Key Insights

Tree-based models are highly effective for tabular medical data

Deep learning does not always outperform classical ML for structured datasets

Proper preprocessing and evaluation metrics are crucial for healthcare applications

📦 Technologies Used

Python

Scikit-learn

XGBoost

PyTorch

Pandas, NumPy

Matplotlib, Seaborn
