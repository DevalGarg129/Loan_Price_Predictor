# Loan Approval Prediction

## Overview
This project loads the loan approval dataset, preprocesses the raw features, and compares several classifiers to predict whether a loan will be approved.

## Features
- Data loading and validation for the loan dataset
- Handling of missing values and target encoding
- Mixed-type preprocessing with numerical scaling and one-hot encoding
- Training and comparison of multiple classifiers
- Accuracy, balanced accuracy, ROC AUC, and confusion-matrix reporting
- CI workflow that runs the project tests automatically

## Tech Stack
- Python 3
- NumPy, Pandas, Matplotlib
- scikit-learn
- GitHub Actions for CI

## CI Pipeline
- Set up the Python environment
- Install project dependencies
- Run the regression tests
- Validate the model training pipeline

## How to Run
```bash
pip install -r requirements.txt
python -m unittest -q
python main.py
```

## Dataset
The project reads the `LoanApprovalPrediction.csv` file and predicts the `Loan_Status` target using features such as income, education, employment, and property area.