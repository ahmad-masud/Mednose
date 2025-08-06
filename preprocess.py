from sklearn.preprocessing import MultiLabelBinarizer


def prepare_symptom_data(df):
    """
    Preprocess the dataset to extract symptoms and encode them for model training.

    This function:
    - Identifies all columns containing symptom data
    - Cleans and compiles the symptoms into a list for each record
    - Uses MultiLabelBinarizer to convert the list of symptoms into a binary feature matrix

    Parameters:
    df (DataFrame): The input DataFrame containing symptom columns and a target 'Disease' column.

    Returns:
    X (ndarray): Binary matrix representing the presence of symptoms.
    y (Series): Target labels (disease names).
    mlb (MultiLabelBinarizer): Fitted binarizer used to transform and inverse-transform symptom data.
    """
    symptom_cols = [col for col in df.columns if "Symptom" in col]
    df[symptom_cols] = df[symptom_cols].fillna("")
    df["symptom_list"] = df[symptom_cols].values.tolist()
    df["symptom_list"] = df["symptom_list"].apply(
        lambda x: [s.strip() for s in x if s.strip()]
    )

    mlb = MultiLabelBinarizer()
    X = mlb.fit_transform(df["symptom_list"])
    y = df["Disease"]
    return X, y, mlb
