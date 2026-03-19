from pathlib import Path
import pandas as pd


def load_customer_data():

    BASE_DIR = Path(__file__).resolve().parent.parent.parent

    data_path = BASE_DIR / "data" / "Mall_Customers.csv"

    df = pd.read_csv(data_path)

    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

    return df, X