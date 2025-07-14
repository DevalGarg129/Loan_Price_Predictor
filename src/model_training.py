from sklearn.ensemble import RandomForestRegressor
import os
import joblib
import yaml

def train_model(X_train, y_train):
    """
    Train a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, scaler, config_path="config.yaml"):
    """
    Save the trained model and scaler to the models/ directory.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model_dir = config.get('model_dir', 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'final_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"✅ Model saved at: {model_path}")
    print(f"✅ Scaler saved at: {scaler_path}")
