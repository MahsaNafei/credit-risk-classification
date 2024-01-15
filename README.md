## credit-risk-classification

### Overview of the Analysis

In this comprehensive analysis, the primary objective was to develop and assess a machine learning model capable of predicting loan statuses based on financial data. The dataset, sourced from lending activities, contained vital information crucial for predicting whether a loan is classified as healthy (0) or high-risk (1). This analysis aimed to employ logistic regression models and explore the impact of resampling techniques on predictive performance.

#### Data Exploration and Preparation

The initial steps involved reading the lending data from the 'Resources' folder into a Pandas DataFrame. The dataset was then split into labels ('y') derived from the 'loan_status' column and features ('X') obtained from the remaining columns. A critical check for label balance using the `value_counts` function revealed insights into the distribution of healthy and high-risk loans.

#### Logistic Regression with Original Data

The first phase of model development utilized a logistic regression model trained with the original dataset. The 'lending_data.csv' was split into training and testing sets using the `train_test_split` method. Subsequently, the logistic regression model was instantiated, fitted using the training data, and evaluated for performance. Key metrics such as balanced accuracy, precision, recall, F1-score, and support were computed to gauge the model's predictive capabilities.

#### Logistic Regression with Resampled Data

In the second phase, the analysis explored the impact of oversampling on model performance. The `RandomOverSampler` module from the imbalanced-learn library was employed to balance the number of data points for each label, thus addressing the class imbalance issue. The logistic regression model was then fitted using the resampled training data, and its predictive performance was assessed in a manner similar to the original model.

#### Key Metrics and Performance Evaluation

Throughout the analysis, essential metrics such as balanced accuracy, precision, recall, and F1-score were computed to quantify the model's predictive accuracy, particularly its ability to identify high-risk loans. The confusion matrix and classification report provided detailed insights into the model's true positives, true negatives, false positives, and false negatives.

### Results

#### Machine Learning Model 1: Logistic Regression with Original Data

- **Balanced Accuracy Score:** 0.99
- **Precision (Label 0):** 1.00
- **Precision (Label 1):** 0.87
- **Recall (Label 0):** 1.00
- **Recall (Label 1):** 0.89
- **F1-Score (Label 0):** 1.00
- **F1-Score (Label 1):** 0.93
- **Support (Label 0):** 18759
- **Support (Label 1):** 625
- **Accuracy:** 99%

#### Machine Learning Model 2: Logistic Regression with Resampled Data

- **Balanced Accuracy Score:** 1.00
- **Precision (Label 0):** 1.00
- **Precision (Label 1):** 0.87
- **Recall (Label 0):** 1.00
- **Recall (Label 1):** 0.99
- **F1-Score (Label 0):** 1.00
- **F1-Score (Label 1):** 0.93
- **Support (Label 0):** 18759
- **Support (Label 1):** 625
- **Accuracy:** 100%

### Summary

Both machine learning models performed exceptionally well, with high precision, recall, and F1-scores. The logistic regression model trained with oversampled data demonstrated slightly improved performance, achieving a balanced accuracy score of 1.00 and perfect precision and recall for label 0. Considering the significance of correctly identifying high-risk loans (label 1), the model's enhanced performance with oversampled data makes it the preferred choice. The balanced accuracy score of 1.00 indicates a near-perfect classification, making it suitable for predicting loan health status across the entire dataset.

### Data Source

- The lending data for this analysis was sourced from the 'Resources' folder, specifically the 'lending_data.csv' file.
- The dataset includes financial information relevant to loan statuses, with the 'loan_status' column indicating healthy (0) or high-risk (1) loans.
- The raw data file is accessible via the provided relative path: `Resources/lending_data.csv`.

### Instructions

1. Ensure that the necessary Python libraries, including Pandas, NumPy, and scikit-learn, are installed in your Python environment.
2. Open the Jupyter Notebook (`loan_status_prediction.ipynb`) in your local development environment or preferred platform supporting Jupyter Notebooks.
3. Execute the notebook cells sequentially to perform data exploration, model training, and evaluation.
