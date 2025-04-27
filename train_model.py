import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load heart disease dataset
# You must have 'heart.csv' in the same folder
data = pd.read_csv('heart.csv')

# Define features and target
X = data.drop('target', axis=1)  # Assuming 'target' is your label column
y = data['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'heart_disease_model.pkl')

print("✅ Heart disease model trained and saved as 'heart_disease_model.pkl'")


