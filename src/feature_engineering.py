def add_features(df):
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['EMI'] = df['LoanAmount'] / df['Loan_Amount_Term']
    df['Balance_Income'] = df['Total_Income'] - (df['EMI'] * 1000)
    return df