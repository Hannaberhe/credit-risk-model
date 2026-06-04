import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
rfm = pd.read_csv('data/processed/rfm_with_risk.csv')
features = ['Recency', 'Frequency', 'Monetary']
X = rfm[features]
y = rfm['is_high_risk']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Logistic Regression with GridSearchCV...")
lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=3, scoring='roc_auc')
lr_grid.fit(X_train, y_train)
print(f"Best LR params: {lr_grid.best_params_}")

lr_best = lr_grid.best_estimator_
lr_pred = lr_best.predict(X_test)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, lr_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, lr_best.predict_proba(X_test)[:,1]):.4f}")

print("\nTraining Random Forest with GridSearchCV...")
rf_params = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='roc_auc')
rf_grid.fit(X_train, y_train)
print(f"Best RF params: {rf_grid.best_params_}")

rf_best = rf_grid.best_estimator_
rf_pred = rf_best.predict(X_test)

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, rf_best.predict_proba(X_test)[:,1]):.4f}")

joblib.dump(rf_best, 'models/best_model.pkl')
print("\nBest model saved!")

# MLflow tracking
try:
    import mlflow
    import mlflow.sklearn
    
    mlflow.set_experiment("credit-risk-model")
    
    with mlflow.start_run(run_name="random_forest_gridsearch"):
        mlflow.log_params(rf_grid.best_params_)
        mlflow.log_metric("accuracy", accuracy_score(y_test, rf_pred))
        mlflow.log_metric("precision", precision_score(y_test, rf_pred))
        mlflow.log_metric("recall", recall_score(y_test, rf_pred))
        mlflow.log_metric("f1", f1_score(y_test, rf_pred))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, rf_best.predict_proba(X_test)[:,1]))
        mlflow.sklearn.log_model(rf_best, "model")
        mlflow.register_model("runs:/" + mlflow.active_run().info.run_id + "/model", "credit-risk-rf")
        print("MLflow run logged and model registered!")
except ImportError:
    print("MLflow not installed. Skipping experiment tracking.")

print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)
print(f"Logistic Regression  Accuracy: {accuracy_score(y_test, lr_pred):.4f}  ROC-AUC: {roc_auc_score(y_test, lr_best.predict_proba(X_test)[:,1]):.4f}")
print(f"Random Forest        Accuracy: {accuracy_score(y_test, rf_pred):.4f}  ROC-AUC: {roc_auc_score(y_test, rf_best.predict_proba(X_test)[:,1]):.4f}")
print("="*50)
