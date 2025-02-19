# credit_default_prediction
# Predicting Credit Card Defaults using Supervised Machine Learning Models

## Project Structure
credit_default.ipynb contains all data cleaning, pre-processing, modelling and results. <br>
model_comparison contains final model results and graphs. <br>
credit_default_presentation contains further written detail on the project.

## Context
Create a classification algorithm to predict whether a customer will default on their credit card. 
Enabling financial institutions to use machine learning, rather than relying on credit scores in customer approval.

## Summary of Findings

### Data Summary
- Dataset contained 30,000 rows and 24 columns.
- Target feature was 'defaulter' (Formerly named default.payment.next.month).

### Data Pre-processing:
- SMOTE applied to dataset. Both normal and resampled datasets used in modelling.
- Train/test split used.
- Continious features scaled.

### Data Insights:
- Data imbalance of 22% for the target feature, defaulters.
- Most recent repayment status was the most important feature.


### Modelling Results
#### Models used:
1. Support Vector Machine (SVM)
2. Random Forest
3. K-Nearest Neighbour
4. XGBoost
5. Logistic Regression

#### Evaluation metrics:
- Accuracy, precision, recall and F1-score. Focus on F1 score with higher recall due to aim in classifying the most defaulters. Precision also considered not inhibit overall client quantity.

#### Findings
- XGBoost achieved the highest F1-score-recall combination after hyperparameter tuning, also one of the quickest models to tune.
- XGBoost predicted 575 additional actual defaulters than the Dummy Classifer.
- Higher education reduces liklihood of the customer defaulting according to model SHAP analysis.

### Next Steps
1. Explore feature engineering by focusing on most influentual features.
2. Hyperparameter tuning of SVM model.
3. Attempt other resampling techniques to address data imbalance.
