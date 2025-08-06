import kagglehub
import pandas as pd


def load_dataset():
    """
    Download and load the disease and symptoms dataset from Kaggle.

    This function uses kagglehub to download the dataset and reads two CSV files:
    - 'DiseaseAndSymptoms.csv': Contains diseases and associated symptoms.
    - 'Disease precaution.csv': Contains precautionary measures for diseases.

    Returns:
    df (DataFrame): DataFrame containing diseases and symptoms.
    precaution_df (DataFrame): DataFrame containing disease precautions.
    """
    path = kagglehub.dataset_download("choongqianzheng/disease-and-symptoms-dataset")
    df = pd.read_csv(f"{path}/DiseaseAndSymptoms.csv")
    precaution_df = pd.read_csv(f"{path}/Disease precaution.csv")
    return df, precaution_df
