import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.data_processing import calculate_rfm, create_aggregate_features

def test_aggregate_features():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300],
        'TransactionStartTime': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']),
        'TransactionId': [1, 2, 3]
    })
    agg = create_aggregate_features(df)
    assert len(agg) == 2
    assert agg[agg['CustomerId'] == 1]['total_amount'].values[0] == 300
    assert agg[agg['CustomerId'] == 1]['transaction_count'].values[0] == 2

def test_rfm_calculation():
    df = pd.DataFrame({
        'CustomerId': [1, 1, 2],
        'Amount': [100, 200, 300],
        'TransactionStartTime': pd.to_datetime(['2024-01-01', '2024-06-01', '2024-01-03']),
        'TransactionId': [1, 2, 3]
    })
    rfm = calculate_rfm(df)
    assert len(rfm) == 2
    assert 'Recency' in rfm.columns
    assert 'Frequency' in rfm.columns
    assert 'Monetary' in rfm.columns
