import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# -----------------------------
# 1. Load Data
# -----------------------------
df = pd.read_csv("BankChurners.csv")

# Drop ID column
df.drop(columns=["CLIENTNUM"], inplace=True)

# Encode target
df["Attrition_Flag"] = df["Attrition_Flag"].map({
    "Existing Customer": 0,
    "Attrited Customer": 1
})

# -----------------------------
# 2. Feature Selection
# -----------------------------
features = [
    'Customer_Age', 'Gender', 'Dependent_count', 'Education_Level',
    'Marital_Status', 'Income_Category', 'Card_Category',
    'Months_on_book', 'Total_Relationship_Count',
    'Months_Inactive_12_mon', 'Contacts_Count_12_mon',
    'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

X = df[features]
y = df["Attrition_Flag"]

# -----------------------------
# 3. Split FIRST (IMPORTANT)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 4. Preprocessing
# -----------------------------
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(include="number").columns

# Encode categoricals
encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_train_cat = encoder.fit_transform(X_train[categorical_cols])
X_test_cat = encoder.transform(X_test[categorical_cols])

# Scale numericals
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical_cols])
X_test_num = scaler.transform(X_test[numerical_cols])

# Combine
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# -----------------------------
# 5. Handle Imbalance (TRAIN ONLY)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

# -----------------------------
# 6. Train Model
# -----------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train_bal, y_train_bal)

# -----------------------------
# 7. Cross-Validation
# -----------------------------
cv_scores = cross_val_score(
    model,
    X_train_bal,
    y_train_bal,
    cv=5,
    scoring="roc_auc"
)

print(f"Cross-Validation AUC: {cv_scores.mean():.3f}")

# -----------------------------
# 8. Evaluation
# -----------------------------
y_pred = model.predict(X_test_processed)
y_pred_proba = model.predict_proba(X_test_processed)[:, 1]

print(f"Test ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -----------------------------
# 9. ROC Curve
# -----------------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(7,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_pred_proba):.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Customer Churn Prediction")
plt.legend()
#plt.show()

# -----------------------------
# 10. Feature Importance
# -----------------------------
feature_names = (
    list(numerical_cols) +
    list(encoder.get_feature_names_out(categorical_cols))
)

importances = model.feature_importances_
fi = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(fi.head(10))

# -----------------------------
# 11. Save Model
# -----------------------------

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoder, "encoder.pkl")
print("Model, scaler, and encoder saved successfully.")

print("🔥 I REACHED THE END OF THE FILE 🔥")