import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def load_and_preprocess_data(filepath="LoanApprovalPrediction.csv"):
    data = pd.read_csv(filepath)

    if "Loan_Status" not in data.columns:
        raise ValueError("Target variable 'Loan_Status' is missing from the dataset.")

    data = data.drop(columns=["Loan_ID"], errors="ignore")
    data = data.dropna(subset=["Loan_Status"]).reset_index(drop=True)

    X = data.drop(columns=["Loan_Status"])
    y = data["Loan_Status"].map({"Y": 1, "N": 0})

    if y.isna().any():
        raise ValueError("Target variable contains unexpected values. Expected 'Y' or 'N'.")

    return X, y.astype(int)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []

    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVC": SVC(probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=5000, random_state=42),
    }

    results = {}

    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            "accuracy": float(accuracy),
            "balanced_accuracy": float(balanced_accuracy),
            "roc_auc": float(roc_auc) if roc_auc is not None else None,
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

        print(f"{name} Accuracy: {accuracy * 100:.2f}%")

    return results


if __name__ == "__main__":
    print("🚀 Starting Loan Approval Prediction Pipeline")

    X, y = load_and_preprocess_data()
    results = train_and_evaluate(X, y)

    best_model_name, best_model_metrics = max(
        results.items(),
        key=lambda item: item[1]["accuracy"],
    )

    print(f"Best model: {best_model_name} with accuracy {best_model_metrics['accuracy'] * 100:.2f}%")
    print("✅ Pipeline completed successfully")
