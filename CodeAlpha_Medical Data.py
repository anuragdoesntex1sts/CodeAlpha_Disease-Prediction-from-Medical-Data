import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------------------------------------
# 1. LOAD THE MEDICAL DATASET
# ---------------------------------------------------------
# Loading the UCI Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
# 0 = Malignant (Disease), 1 = Benign (No Disease)
# We will invert this so 1 = Disease (Positive Class) for standard medical evaluation
y = 1 - data.target 

print(f"Dataset loaded: {X.shape[0]} patients, {X.shape[1]} clinical features.")

# ---------------------------------------------------------
# 2. DATA PREPROCESSING
# ---------------------------------------------------------
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling: Vital for Logistic Regression and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------
# 3. DEFINE THE ALGORITHMS
# ---------------------------------------------------------
# We use standard hyperparameters for baseline comparison
models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Support Vector Machine (SVM)": SVC(kernel='rbf', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ---------------------------------------------------------
# 4. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------
results = []

for name, model in models.items():
    # Train the model
    # Tree-based models (RF, XGBoost) don't strictly need scaling, but it doesn't hurt them.
    # Linear/Distance models (LR, SVM) absolutely require scaling.
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the unseen test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate performance metrics
    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred), 3),
        "Recall": round(recall_score(y_test, y_pred), 3),
        "F1-Score": round(f1_score(y_test, y_pred), 3)
    })

# ---------------------------------------------------------
# 5. DISPLAY RESULTS
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n=== Medical Disease Prediction Model Evaluation ===")
print(results_df.to_string(index=False))