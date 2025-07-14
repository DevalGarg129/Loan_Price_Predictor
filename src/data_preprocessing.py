import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df['source'] = 'train'
    test_df['source'] = 'test'

    df = pd.concat([train_df, test_df], ignore_index=True)

    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

    return df


def encode_categoricals(df):
    cols = ['Gender','Married','Education','Self_Employed','Property_Area','Dependents']
    for col in cols:
        df[col] = LabelEncoder().fit_transform(df[col])
    return df
