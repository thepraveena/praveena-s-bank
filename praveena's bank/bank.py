import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Setup and Synthetic Data Generation
np.random.seed(42)
n_application = 1000

data = {
    'credit_score': np.random.randint(300, 900, n_application),
    'debt_to_income_ratio': np.random.uniform(0.1, 0.9, n_application),
    'missed_payment_count': np.random.randint(0, 6, n_application),
    'requested_amount_lakhs': np.random.uniform(1, 50, n_application)
}
df = pd.DataFrame(data)

# Define target labels based on risk conditions
df['is_high_risk_fraud'] = (
    (df['credit_score'] < 450) & 
    (df['missed_payment_count'] > 3) & 
    (df['requested_amount_lakhs'] > 20)
).astype(int)

# 2. Feature Selection and Train-Test Split
features = ['credit_score', 'debt_to_income_ratio', 'missed_payment_count', 'requested_amount_lakhs']
X = df[features]
y = df['is_high_risk_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Preprocessing (Scaling & PCA)
# Wrapping the input in a DataFrame helps avoid feature name warnings during inference
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# 4. Model Training and Evaluation
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_pca, y_train)

y_pred = model.predict(X_test_pca)
print(f"Bank risk detection accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

# 5. Live Application Inference
# Passing as a DataFrame with matching feature names to cleanly suppress the UserWarning
new_application = pd.DataFrame([[200, 0.85, 100, 35.0]], columns=features)

new_scaled = scaler.transform(new_application)
new_pca = pca.transform(new_scaled)
result = model.predict(new_pca)

print("---- live bank application scan solution -----")
print(f"Bank history: {new_application.values[0].tolist()}")

if result[0] == 1:
    print("Alert: high risk/fraud pattern detected. Reject loan application.")
else:
    print("Success: safe profile approved. Processing loan verification.") 