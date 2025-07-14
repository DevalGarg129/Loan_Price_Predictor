import pandas as pd
import yaml
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_preprocessing import load_and_preprocess_data, encode_categoricals
from src.feature_engineering import add_features
from src.utils import scale_features
from src.model_training import train_model, save_model
from src.evaluation import evaluate_model

# === Load config ===
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# === Load and preprocess data ===
df = load_and_preprocess_data(config['data']['train_path'], config['data']['test_path'])
df = encode_categoricals(df)
df = add_features(df)

# === Separate back ===
train_df = df[df['source'] == 'train'].copy()
test_df = df[df['source'] == 'test'].copy()

# === Prepare data ===
X = train_df.drop(columns=["Loan_ID", "Loan_Status", "source"])
y = train_df["LoanAmount"]
X_test_final = test_df.drop(columns=["Loan_ID", "Loan_Status", "source"], errors='ignore')

# === Train-validation split ===
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=config['training']['test_size'],
    random_state=config['training']['random_state']
)

# === Scale features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# === Train model ===
model = train_model(X_train_scaled, y_train)

# === Evaluate ===
y_val_pred = model.predict(X_val_scaled)
mse, r2 = evaluate_model(y_val, y_val_pred)
print(f"✅ Validation MSE: {mse:.2f}")
print(f"✅ Validation R² Score: {r2:.2f}")

# === Save model and scaler ===
save_model(model, scaler, config_path="config.yaml")

# === Predict on test set ===
test_preds = model.predict(X_test_scaled)

# === Save predictions to outputs ===
os.makedirs("outputs", exist_ok=True)
submission = pd.DataFrame({
    "Loan_ID": test_df["Loan_ID"].values,
    "Predicted_LoanAmount": test_preds
})
submission_path = os.path.join("outputs", "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"📦 Submission file saved as {submission_path}")

# === Save metrics ===
metrics = {
    "mse": round(mse, 2),
    "r2_score": round(r2, 2)
}
os.makedirs("outputs/metrics", exist_ok=True)
with open("outputs/metrics/metrics.json", "w") as f:
    yaml.dump(metrics, f)
print("📊 Metrics saved to outputs/metrics/metrics.json")
