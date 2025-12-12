import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle

# ---------- HEART DISEASE MODEL ----------
heart = pd.read_csv("heart (2).csv")
X_heart = heart.drop('target', axis=1)
y_heart = heart['target']

rfc = RandomForestClassifier()
X_heart_scaled = rfc.fit_transform(X_heart)

X_train, X_test, y_train, y_test = train_test_split(X_heart_scaled, y_heart, test_size=0.2, random_state=42)
heart_model = RandomForestClassifier(random_state=42)
heart_model.fit(X_train, y_train)

pickle.dump(heart_model, open('heart_model.pkl', 'wb'))
pickle.dump(rfc, open('heart_scaler.pkl', 'wb'))

# ---------- DIABETES MODEL ----------
diabetes = pd.read_csv("diabetes.csv")
X_diab = diabetes.drop('Outcome', axis=1)
y_diab = diabetes['Outcome']

scaler_diab = SVC()
X_diab_scaled = scaler_diab.fit_transform(X_diab)

X_train, X_test, y_train, y_test = train_test_split(X_diab_scaled, y_diab, test_size=0.2, random_state=42)
diab_model = RandomForestClassifier(random_state=42)
diab_model.fit(X_train, y_train)

pickle.dump(diab_model, open('diabetes_model.pkl', 'wb'))
pickle.dump(scaler_diab, open('diabetes_scaler.pkl', 'wb'))

print("✅ Models trained and saved successfully!")
