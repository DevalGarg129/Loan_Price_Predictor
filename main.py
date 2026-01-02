import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


def load_and_preprocess_data():
    data = pd.read_csv("LoanApprovalPrediction.csv")

    # Drop unique ID column
    if "Loan_ID" in data.columns:
        data = data.drop("Loan_ID", axis=1)

    # Encode categorical variables
    label_encoder = preprocessing.LabelEncoder()
    for col in data.select_dtypes(include="object").columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Fill missing values with median
    for col in data.columns:
        data[col] = data[col].fillna(data[col].median())

    return data


def train_and_evaluate(data):
    X = data.drop("Loan_Status", axis=1)
    Y = data["Loan_Status"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))]),
        "SVC": Pipeline([("scaler", StandardScaler()), ("clf", SVC())]),
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))])
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        acc = metrics.accuracy_score(Y_test, y_pred)
        results[name] = acc
        print(f"{name} Accuracy: {acc * 100:.2f}%")

    return results


if __name__ == "__main__":
    print("🚀 Starting Loan Approval Prediction Pipeline")

    data = load_and_preprocess_data()
    results = train_and_evaluate(data)

    print("✅ Pipeline completed successfully")
