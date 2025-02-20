# credit_default_prediction
# Predicting Credit Card Defaults using Supervised Machine Learning Models

## Project Structure
credit_default.ipynb contains data cleaning, pre-processing, modelling, graphs and results. <br>
  - NOTE: Cells containing cross-validation are converted to markdown format. The original outputs are often saved in the next cell under `best_params`. <br>
  
model_comparison contains the final model results and graphs. <br>
credit_default_presentation contains further written details on the project.

## Context
Create a classification algorithm to predict whether customers will default on their credit cards. 
Enabling financial institutions to use machine learning, rather than relying on credit scores in customer approval.

## Summary of Findings

### Data Summary
- Dataset contains 30,000 rows and 24 columns.
- The target feature is 'defaulter' (Formerly named default.payment.next.month).

### Data Pre-processing:
- SMOTE applied to the dataset. Both normal and resampled datasets are used in modelling.
- Train/test split used.
- Continuous features scaled.

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
- Accuracy, precision, recall and F1-score. Focus on F1 score with higher recall due to aim in classifying the most defaulters. Precision is also considered to improve customer revenue by not rejecting actual non-defaulters.

#### Findings
- XGBoost achieved the highest F1-score-recall (0.53 and 0.56) combination after hyperparameter tuning and one of the quickest models to tune.
- XGBoost predicted 575 additional actual defaulters than the Dummy Classifer.
- Higher education reduces the likelihood of customers defaulting on credit cards.

### Next Steps
1. Explore feature engineering by focusing on the most influential features.
2. Hyperparameter tuning of the SVM model. Although the run time was among the longest without tuning.
3. Attempt other resampling techniques to address data imbalance.
