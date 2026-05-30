# Credit Risk Model - Interim Report
**Hanna Berhe | May 31, 2026**

## Business Understanding
Bati Bank is building a buy-now-pay-later service. I am creating a credit scoring model using transaction data from an eCommerce partner. Since there is no default label in the data, I will use customer behavior (RFM analysis) to create a proxy for credit risk.

## EDA Findings
- Dataset: 95,662 transactions, 16 columns
- No missing values - clean data
- Fraud rate: 0.2% (193 cases)
- Top category: Financial services (47%)
- Top channel: Channel 3 (59%)
- Transaction amounts vary widely from negative to 9.8M

## Charts Created
- Transaction amount distribution
- Fraud vs non-fraud comparison
- Top product categories
- Amount by channel

## Next Steps
- Feature engineering (RFM metrics)
- K-Means clustering for risk labels
- Model training with MLflow
- API deployment

## GitHub
https://github.com/Hannaberhe/credit-risk-model
