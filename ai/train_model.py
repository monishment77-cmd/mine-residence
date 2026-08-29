import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
data_path = "data/raw_subsidence_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"'{data_path}' not found. Run Step 1 (generate_data.py) first.")

print("Loading dataset...")
df = pd.read_csv(data_path)

# 2. Select Input Features & Target Label
feature_cols = [
    'tilt_x', 'tilt_y', 'total_tilt', 'vibration', 'displacement',
    'crack_level', 'temperature', 'displacement_rate', 'tilt_rate', 'vibration_change'
]
target_col = 'label'

X = df[feature_cols]
y = df[target_col]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Isolation Forest (Higher sensitivity so all failure modes flag DETECTED)
print("\n--- Training Isolation Forest (Anomaly Detection) ---")
# Increased contamination to 0.18 to capture Subsurface Failure & Shear Cracks
iso_forest = IsolationForest(n_estimators=100, contamination=0.18, random_state=42)
iso_forest.fit(X_train_scaled)

# Test Anomaly Detection (-1 = Anomaly, 1 = Normal)
anomaly_preds = iso_forest.predict(X_test_scaled)
anomalies_detected = (anomaly_preds == -1).sum()
print(f"Isolation Forest identified {anomalies_detected} anomalies out of {len(X_test)} test records.")

# 5. Train Supervised Classifier (Random Forest)
print("\n--- Training Random Forest Classifier (Risk Classification) ---")
rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# 6. Model Evaluation
y_pred = rf_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=["NORMAL", "WATCH", "WARNING", "CRITICAL"]))

# 7. Feature Importance Analysis
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nFeature Importances:")
print(importance_df.to_string(index=False))

# 8. Risk Score Calculator
def calculate_risk_score(model, feature_vector):
    probs = model.predict_proba(feature_vector)[0]
    classes = list(model.classes_)
    
    prob_dict = {cls: prob for cls, prob in zip(classes, probs)}
    
    score = (
        prob_dict.get('NORMAL', 0) * 15 +
        prob_dict.get('WATCH', 0) * 45 +
        prob_dict.get('WARNING', 0) * 75 +
        prob_dict.get('CRITICAL', 0) * 95
    )
    return round(float(score), 2)

# 9. Save Models & Scaler to `models/` folder
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/risk_classifier.pkl")
joblib.dump(iso_forest, "models/anomaly_detector.pkl")
joblib.dump(scaler, "models/feature_scaler.pkl")
joblib.dump(feature_cols, "models/feature_columns.pkl")

print("\nAll models, scalers, and metadata saved to 'models/' directory successfully.")