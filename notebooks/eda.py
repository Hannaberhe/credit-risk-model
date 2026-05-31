import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/raw/data.csv')

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nColumn names:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic stats:")
print(df.describe())
print("\nTop categories:")
print(df['ProductCategory'].value_counts().head(10))
print("\nTop channels:")
print(df['ChannelId'].value_counts().head(10))
print("\nFraud count:")
print(df['FraudResult'].value_counts())

# Chart: Amount distribution
df['Amount'].hist(bins=50, color='steelblue')
plt.title('Transaction Amount Distribution')
plt.xlabel('Amount')
plt.savefig('reports/amount_chart.png', dpi=100)
print("\nChart saved!")

print("\nDone!")

# More charts
import matplotlib.pyplot as plt

# Chart 2: Fraud vs Non-Fraud
df['FraudResult'].value_counts().plot(kind='bar', color=['green', 'red'])
plt.title('Fraud vs Non-Fraud Transactions')
plt.xlabel('Fraud (1=Yes, 0=No)')
plt.savefig('reports/fraud_chart.png', dpi=100)
print("Fraud chart saved!")

# Chart 3: Top 5 categories
top5 = df['ProductCategory'].value_counts().head(5)
top5.plot(kind='barh', color='teal')
plt.title('Top 5 Product Categories')
plt.xlabel('Count')
plt.tight_layout()
plt.savefig('reports/category_chart.png', dpi=100)
print("Category chart saved!")

# Chart 4: Amount by Channel
df.boxplot(column='Amount', by='ChannelId')
plt.title('Amount by Channel')
plt.suptitle('')
plt.savefig('reports/channel_chart.png', dpi=100)
print("Channel chart saved!")

print("\n=== TOP 5 INSIGHTS ===")


plt.suptitle('')
plt.savefig('reports/channel_chart.png', dpi=100)
print("Channel chart saved!")

print("\n=== TOP 5 INSIGHTS ===")
print("1. Data has 95,662 transactions with zero missing values")
print("2. Fraud rate is only 0.2% - very rare")
print("3. Financial services is the most popular category (47%)")
print("4. Channel 3 dominates with 59% of transactions")
print("5. Transaction amounts vary widely - some negative (refunds)")

# Correlation matrix
import seaborn as sns
num_cols = ['Amount', 'Value', 'PricingStrategy', 'FraudResult']
corr = df[num_cols].corr()
print("\nCorrelation matrix:")
print(corr)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
ax.set_title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('reports/correlation_chart.png', dpi=100)
print("Correlation chart saved!")

# Box plot for outliers
fig, ax = plt.subplots(figsize=(10, 5))
df.boxplot(column='Amount', ax=ax)
ax.set_title('Box Plot of Transaction Amounts (Outlier Detection)')
ax.set_ylabel('Amount')
plt.tight_layout()
plt.savefig('reports/boxplot_chart.png', dpi=100)
print("Box plot saved!")

print("\nDone with all EDA!")
