# Credit Risk Model - Bati Bank

## Credit Scoring Business Understanding

### 1. Why Does Basel II Require Interpretable Models?

The Basel II Accord is a set of banking regulations. It says banks must measure and document their credit risk. If a bank cannot explain how it makes lending decisions, it can be fined. So our model must be interpretable. We need to show regulators exactly why a customer was approved or denied credit. A model that is a black box is not acceptable.

### 2. Why Do We Need a Proxy Variable?

The data we have is transaction data from an eCommerce platform. There is no column that says "this customer defaulted." So we have to create our own label. We will use customer behavior to decide who is high risk. For example, customers who rarely buy, spend very little, or have not transacted in a long time can be labeled as high risk. This is called a proxy variable. The risk is that our proxy may be wrong. We might label a good customer as bad, or miss a truly risky customer. We must be honest about this limitation.

### 3. Simple vs Complex Models

A simple model like Logistic Regression is easy to explain. We can say "this customer got a low score because of X and Y." Regulators like this. A complex model like XGBoost may predict better but is harder to explain. In banking, being able to explain your decision often matters more than a small improvement in accuracy. We will try both and compare.

## EDA Key Insights
1. 95,662 transactions with zero missing values
2. Financial services and airtime are 95% of transactions
3. Channel 3 dominates with 59% of all transactions
4. Fraud is very rare at only 0.2%
5. Amount and Value are 99% correlated
6. Transaction amounts are highly skewed with outliers
