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

def calculate_rfm(df, snapshot_date=None):
    """Calculate Recency, Frequency, Monetary per customer."""
    if snapshot_date is None:
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerId').agg(
        Recency=('TransactionStartTime', lambda x: (snapshot_date - x.max()).days),
        Frequency=('TransactionId', 'count'),
        Monetary=('Amount', 'sum')
    ).reset_index()
    
    return rfm

def create_risk_label(df, rfm, random_state=42):
    """Create is_high_risk using K-Means clustering on RFM."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    features = ['Recency', 'Frequency', 'Monetary']
    X = rfm[features].copy()
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Cluster into 3 groups
    kmeans = KMeans(n_clusters=3, random_state=random_state, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Find highest risk cluster (lowest Frequency and Monetary)
    cluster_stats = rfm.groupby('cluster')[['Frequency', 'Monetary']].mean()
    high_risk_cluster = cluster_stats['Frequency'].idxmin()
    
    # Create label
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
    
    print(f"Cluster stats:\n{cluster_stats}")
    print(f"High risk cluster: {high_risk_cluster}")
    print(f"High risk customers: {rfm['is_high_risk'].sum()}")
    print(f"Low risk customers: {(rfm['is_high_risk'] == 0).sum()}")
    
    return rfm

if __name__ == "__main__":
    df, agg = process_data()
    print("\nCalculating RFM...")
    rfm = calculate_rfm(df)
    print("\nCreating risk labels...")
    rfm = create_risk_label(df, rfm)
    rfm.to_csv('data/processed/rfm_with_risk.csv', index=False)
    print("\nSaved to data/processed/")
