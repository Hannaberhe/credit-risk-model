import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

def load_data(filepath='data/raw/data.csv'):
    df = pd.read_csv(filepath)
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    return df

def create_aggregate_features(df):
    agg = df.groupby('CustomerId').agg(
        total_amount=('Amount', 'sum'),
        avg_amount=('Amount', 'mean'),
        transaction_count=('Amount', 'count'),
        std_amount=('Amount', 'std')
    ).reset_index()
    agg['std_amount'] = agg['std_amount'].fillna(0)
    return agg

def extract_time_features(df):
    df = df.copy()
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    df['month'] = df['TransactionStartTime'].dt.month
    df['year'] = df['TransactionStartTime'].dt.year
    return df

def process_data(filepath='data/raw/data.csv'):
    print("Loading data...")
    df = load_data(filepath)
    
    print("Creating aggregate features...")
    agg = create_aggregate_features(df)
    
    print("Extracting time features...")
    df = extract_time_features(df)
    
    print("Done!")
    return df, agg

if __name__ == "__main__":
    df, agg = process_data()
    print(f"Transactions: {df.shape}")
    print(f"Aggregated: {agg.shape}")
    agg.to_csv('data/processed/aggregate_features.csv', index=False)
    print("Saved to data/processed/")
