Loan Approval Prediction Model
==============================

This project implements an end-to-end workflow for predicting loan default probability using XGBoost with Optuna-based hyperparameter tuning. It includes data loading, preprocessing, encoding, visualization, model optimization, evaluation, and creation of a Kaggle submission file.

Dataset
-------
The data comes from the Kaggle competition:
https://www.kaggle.com/competitions/loan-approval-prediction

Two CSV files are used:
- train.csv — includes features and the target `loan_status`
- test.csv — includes the same features without labels

Preprocessing
-------------
Key steps:
- Load and inspect train/test data
- Merge datasets temporarily to ensure consistent transformations
- One-hot encode: `person_home_ownership`, `cb_person_default_on_file`
- Map categorical variables:
  - `loan_grade` (A–G → 1–7)
  - `loan_intent` mapped to observed default-rate statistics
- Produce final numeric feature sets: X_train, y_train, X_test

Augmentation
------------
A feature-fuzziness augmentation step optionally perturbs two randomly chosen features per sample. Optuna tunes:
- augProb — probability augmentation is applied
- augAmt — magnitude of perturbation

Model Training
--------------
XGBoost is optimized using Optuna over:
- max_depth
- learning_rate
- subsample
- colsample_bytree
- scale_pos_weight (computed for class imbalance)
- augmentation parameters

The best trial parameters are used to train the final model.

Evaluation
----------
Evaluations include:
- ROC-AUC score
- ROC curve (manual + sklearn)
- Confusion matrix (normalized)
- Predicted probability distributions
- Training accuracy

Submission
----------
The final model generates predicted probabilities for the test set.  
A submission file is created in the required Kaggle format:

id,loan_status

This file can be uploaded directly to the competition page.

Dependencies
------------
numpy  
pandas  
matplotlib  
seaborn  
scikit-learn  
xgboost  
optuna  
tqdm  

These can be installed via pip as needed.
