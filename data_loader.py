import kagglehub
import pandas as pd

def load_dataset():
    path = kagglehub.dataset_download("choongqianzheng/disease-and-symptoms-dataset")
    df = pd.read_csv(f"{path}/DiseaseAndSymptoms.csv")
    precaution_df = pd.read_csv(f"{path}/Disease precaution.csv")
    return df, precaution_df
