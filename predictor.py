from sklearn.preprocessing import MultiLabelBinarizer

def prepare_symptom_data(df):
    symptom_cols = [col for col in df.columns if "Symptom" in col]
    df[symptom_cols] = df[symptom_cols].fillna("")
    df["symptom_list"] = df[symptom_cols].values.tolist()
    df["symptom_list"] = df["symptom_list"].apply(lambda x: [s.strip() for s in x if s.strip()])
    
    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["symptom_list"])
    y = df["Disease"]
    return X, y, mlb
