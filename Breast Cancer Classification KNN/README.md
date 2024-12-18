# Breast Cancer Prediction with K-Nearest Neighbors (KNN)

**Breast Cancer Prediction using KNN!** 🎯💉 This project focuses on predicting whether a breast tumor is malignant or benign based on features extracted from breast cancer cell measurements. By utilizing a **K-Nearest Neighbors (KNN)** classifier, I built a model that leverages these features to predict tumor type, using the **Wisconsin Breast Cancer Dataset**.

## Table of Contents 📑

1. [Key Highlights](#key-highlights)
2. [Project Objectives](#project-objectives)
3. [Results](#results)
4. [Technical Skills Demonstrated](#technical-skills-demonstrated)
5. [Tools & Libraries Used](#tools-and-libraries-used)
6. [How to Run](#how-to-run-the-project)
7. [Conclusion](#conclusion)

## Key Highlights

- **Data Preprocessing**: Cleaned the data by removing unnecessary columns and encoding the labels.
- **Feature Scaling**: Applied **Z-score normalization** to ensure all features are on the same scale for the KNN model.
- **Dimensionality Reduction**: Utilized **Principal Component Analysis (PCA)** to reduce the number of features while retaining variance in the dataset.
- **Model Building**: Built a **K-Nearest Neighbors classifier** with hyperparameter tuning using **GridSearchCV**.
- **Model Evaluation**: Evaluated the model's performance with **recall** as the primary metric to ensure accurate detection of malignant tumors.
- **Visualization**: Used PCA and KNN to visualize the decision boundaries for tumor classification.

## Project Objectives

- Build a **KNN classifier** to predict whether a breast tumor is **malignant** or **benign** based on given features.
- Preprocess the data by encoding labels and applying feature scaling techniques.
- Optimize the model using **GridSearchCV** to find the best hyperparameters for KNN.
- Evaluate the model using **recall** as the key metric to minimize false negatives (missed malignant tumors).

## Results

- **Model Performance**: Achieved **95% accuracy** and **90% recall** for identifying malignant tumors, demonstrating the model’s ability to accurately detect cancerous cells.
- **PCA and KNN Visualization**: Visualized the decision boundaries of the KNN classifier using the first two principal components of the data.

## Technical Skills Demonstrated

- **Data Preprocessing**: Handled missing data, label encoding, and class imbalance.
- **Feature Engineering**: Applied **Z-score normalization** and **PCA** for dimensionality reduction.
- **Model Evaluation**: Used **recall** as the primary metric to ensure the model's effectiveness in detecting malignant tumors.
- **Hyperparameter Tuning**: Performed **GridSearchCV** to tune KNN parameters and achieve the best results.
- **Visualization**: Visualized model performance with **PCA** and decision boundaries to show model's effectiveness.

## Tools and Libraries Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**

## How to Run the Project

1. Download the folder containing the project files.
2. Install required libraries using the following command:
3. Run the Jupyter notebook to view the analysis step by step.

## Conclusion

The **Breast Cancer Prediction** model, built using **K-Nearest Neighbors (KNN)**, successfully predicts whether a tumor is malignant or benign with an accuracy of **95%** and a recall of **90%** for malignant tumors. This project demonstrates the ability to preprocess data, optimize models using grid search, and visualize the results effectively. The use of **PCA** for dimensionality reduction and **KNN** for classification makes the approach both quick and reliable, which can be useful in clinical settings for early breast cancer detection.
