# Fraud Detection 🚨💳
In today's society, where everyone makes purcashes and transactions through electronic means, it is increasingly easier for fraudulent activity to go unnoticed. As such, it is imperative that we create methods to **detect** and **prevent** these transactions. This project focuses on using machine learnign techniques to identify fraudulent activities with a high recall score. This approach allows for almost all fraudulent transactions are caught.

## Table of Contents 📑
- [Key Highlights](#key-highlights)
- [Project Objectives](#project-objectives)
- [Results](#results)
- [Technical Skills Demonstrated](#technical-skills-demonstrated)
- [Tools & Libraries Used](#tools-and-libraries-used)
- [How to Run](#how-to-run-the-project)
- [Conclusion](#conclusion)

## Key Highlights
- **Data Preprocessing**: Cleaned and scaled features. Handled class imbalance using SMOTE.
- **Model Selection**: Tested various models including Logistic Regression, XGBoost, Random Forest.
- **Evaluation Focus**: Focused on maximizing **recall** over accuracy due to the high cost of mislabeling fraud as non-fraud.
- **Visualization**: Used confusion matrix and other metrics to evaluate model performance.

## Project Objectives
- Build a machine learning model that accurately detects fraudulent transactions in a highly imbalanced dataset.
- Preprocess transaction data by scaling, encoding, and handling missing values.
- Improve model performance by focusing on recall to capture as many fraudulent transactions as possible.

## Results
- **Recall**: Achieved a recall of **99%**, indicating that the model was able to detect 99% of fraudulent transactions.
- **Confusion Matrix**: The confusion matrix clearly shows that the model successfully identifies fraudulent transactions while maintaining a balance between false positives and true positives.
  
### Model Performance
- The **XGBoost** model showed the highest recall, outperforming other models in terms of identifying fraudulent transactions.
  
## Technical Skills Demonstrated
- **Data Preprocessing**: Handling missing data, scaling features using MinMaxScaler, encoding categorical variables using one-hot encoding, and addressing class imbalance with SMOTE.
- **Modeling**: Implementing multiple machine learning algorithms, including Logistic Regression, XGBoost, and Random Forest.
- **Model Evaluation**: Evaluating models using recall, confusion matrix, and ROC AUC to ensure high performance in detecting fraud.
- **Data Visualization**: Using libraries like Seaborn and Matplotlib to visualize data distributions, feature correlations, and model performance.

## Tools and Libraries Used
- **Python**
- **Pandas**: Data manipulation and preprocessing
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Machine learning algorithms and model evaluation
- **XGBoost**: Gradient boosting algorithm
- **Imbalanced-learn (SMOTE)**: Handling class imbalance

## How to Run the Project
1. Download the project folder.
2. Install the required libraries.
3. Load the dataset and run the Jupyter notebook to view the analysis step-by-step.

## Conclusion
The **Fraud Detection** model built using various machine learning algorithms (Logistic Regression, XGBoost, Random Forest) successfully identifies fraudulent transactions with a recall rate of **99%**. By focusing on recall, the model ensures that fraudulent transactions are caught, making it an necessary tool for financial companies looking to protect their users from fraud. This project demonstrates a deep understanding of machine learning, data preprocessing, and model evaluation, providing a practical solution to real-world challenges in the finance sector.
