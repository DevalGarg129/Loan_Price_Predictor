import joblib
import os 
import yaml
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__),"../config.yaml")

def read_params(config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path):
    config = read_params()
    model_path = config['model_dir']
    model = joblib.load(os.path.join(model_path, "final_model.pkl"))
    return model

def load_scalar():
    config = read_params()
    scalar_path = config["model_dir"]
    scalar = joblib.load(os.path.join(scalar_path, "scalar.pkl"))
    return scalar

def preprocess_input(input_json):
    df = pd.DataFrame(input_json, index=[0])
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df.drop(['Loan_ID'], axis=1, errors="ignore", inplace=True)
    return df