# Wine Clustering Analysis with K-Means

**Wine Clustering Analysis!** 🍷✨ In this project, I used machine learning techniques to cluster wines into distinct groups based on various chemical and sensory properties. Using **K-Means clustering**, I explored how unsupervised learning can uncover hidden patterns and relationships in the wine dataset, without relying on predefined labels.

## Table of Contents 📑

1. [Key Highlights](#key-highlights)
2. [Project Objectives](#project-objectives)
3. [Results](#results)
4. [Technical Skills Demonstrated](#technical-skills-demonstrated)
5. [Tools & Libraries Used](#tools-and-libraries-used)
6. [How to Run](#how-to-run-the-project)
7. [Conclusion](#conclusion)

## Key Highlights

- **Data Preprocessing**: Cleaned and transformed raw data by handling missing values, outliers, and applying feature scaling (Z-score normalization, log transformation).
- **Feature Selection**: Analyzed correlations between features and removed redundant ones.
- **Clustering with K-Means**: Applied the **K-Means algorithm** to cluster wine data into two groups (Red and White wine). Used the **Elbow Method** to determine the optimal number of clusters.
- **Model Evaluation**: Evaluated clustering quality with **Silhouette Score** and **Adjusted Rand Index (ARI)** to compare predicted clusters with true wine labels.
- **Visualization**: Utilized **t-SNE** to visualize clustering results and correlations.
- **Real-World Application**: Demonstrated the effectiveness of unsupervised learning in grouping wines with similar characteristics, contributing to the understanding of wine classification based on chemical properties.

## Project Objectives

- Identify patterns in wine data through unsupervised learning.
- Apply feature scaling and selection to improve clustering performance.
- Compare clustering results to real-world wine types and evaluate model effectiveness.
- Create clear visualizations that communicate findings effectively.

## Results

- **Cluster Distribution**: The K-Means model identified two clusters, which corresponded well with the actual wine types (Red vs. White).
- **Silhouette Score**: Achieved a score of **0.27**, indicating that clusters are separated but share similar ranges for most of the features.
- **Adjusted Rand Index**: Showed alignment between the predicted clusters and actual wine types, with an ARI score of **0.91**.

## Technical Skills Demonstrated

- **Data Cleaning & Preprocessing**: Handling missing data, outlier detection, feature scaling, log transformation.
- **Feature Engineering**: Correlation analysis for multicollinearity, feature selection, and dimensionality reduction with **t-SNE**.
- **Clustering & Machine Learning**: Implementing K-Means clustering, evaluating models with silhouette score and ARI.
- **Data Visualization**: Visualizing data distributions, correlation matrices, and clustering results with **Matplotlib**, and **Seaborn**.
- **Python & Libraries**: Proficient use of Python with **Pandas**, **NumPy**, **Scikit-learn**, **Matplotlib**, and **Seaborn** for data manipulation, modeling, and visualization.
- **Model Evaluation**: Ability in using clustering evaluation metrics like **Silhouette Score**, **Adjusted Rand Index (ARI)**, and centroid analysis to interpret clustering results.

## Tools and Libraries Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## How to Run the Project

1. Download the folder
2. Install required libraries
3. Run Jupyter notebook to view analysis step by step

# Conclusion

The K-Means clustering model successfully grouped the wine data into two clusters, closely aligning with the actual wine types (red and white). The evaluation metrics, suggest that the model's clustering is quite accurate. However, the low Silhouette Score shows that the clusters may not be separation well between clusters. 

This project demonstrates my ability in data preprocessing, unsupervised learning, and model evaluation. The analysis also highlights my ability to derive meaningful insights from real-world data.