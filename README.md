# Iris Classification

This project implements and evaluates four supervised machine learning models (k-Nearest Neighbours, Naive Bayes, Logistic Regression, and Decision Tree) on the classic Iris dataset. It explores how dimensionality reduction using PCA affects model performance, and analyses model complexity, feature importance, and overfitting tendencies.


## Objectives

- Classify iris flower species (Setosa, Versicolor, Virginica) based on sepal and petal measurements.
- Compare the performance of multiple classifiers on original and PCA-reduced data.
- Perform hyperparameter tuning and model complexity analysis.
- Identify important features contributing to classification.


## Dataset

- **Source**: `sklearn.datasets.load_iris()`
- 150 samples, 4 features:
  - Sepal Length, Sepal Width
  - Petal Length, Petal Width
- 3 target classes: Setosa, Versicolor, Virginica


## Models Implemented

| Model              | Implemented From Scratch | PCA Tested | Notes                           |
|-------------------|---------------------------|------------|---------------------------------|
| k-Nearest Neighbours (kNN) | ✅                         | ✅         | Tuned for `k = 5`               |
| Naive Bayes (Gaussian)     | ✅                         | ✅         | Assumes feature independence    |
| Logistic Regression        | ❌ (Used `sklearn`)        | ✅         | Tuned inverse regularization `C` |
| Decision Tree              | ❌ (Used `sklearn`)        | ✅         | Tuned max depth                 |


## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Feature Importance (Decision Tree, Logistic Regression)
- PCA Explained Variance
- Model Complexity Analysis


## Key Findings

- **All models** achieved **perfect accuracy (1.0)** on the original dataset.
- After PCA (reduced to 2D), performance dropped slightly for all models:
  - kNN: 0.93
  - Naive Bayes: 0.97
  - Logistic Regression: 0.90
  - Decision Tree: 0.87
- **Petal Length** and **Petal Width** were most important for classification.
- PCA preserved ~97% variance, reducing dimensionality with minimal loss in accuracy.
- Overfitting/underfitting was analysed using:
  - `k` for kNN
  - `C` for Logistic Regression
  - `max_depth` for Decision Tree


## Model Complexity Insights

- kNN with small `k` overfits, large `k` underfits. Optimal at **k = 5**.
- Logistic Regression: Stronger regularisation (lower `C`) simplified model.
- Decision Tree: Best depth ≈ **4**, avoiding both over/underfitting.
