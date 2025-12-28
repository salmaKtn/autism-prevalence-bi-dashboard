# Machine Learning - Autism Prevalence Prediction
# Random Forest Regression
# Author: Salma
# Date: December 2024

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

DB_CONFIG = {
    'host': 'localhost',
    'database': 'DWH_ASD',
    'user': 'postgres',
    'password': 'admin'
}

connection_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"

print("=" * 80)
print("MACHINE LEARNING - AUTISM PREVALENCE PREDICTION")
print("=" * 80)

# Load data
print("\nLoading data from PostgreSQL...")

sql_query = """
SELECT 
    f.study_id,
    f.prevalence_per_1000,
    f.sample_size,
    f.male_female_ratio,
    t.year,
    l.country,
    l.is_nationwide,
    m.case_identification_method,
    m.case_criterion,
    d.min_age,
    d.max_age
FROM FACT_ASD_Studies f
JOIN DIM_Time t ON f.time_key = t.time_key
JOIN DIM_Location l ON f.location_key = l.location_key
JOIN DIM_Methodology m ON f.methodology_key = m.methodology_key
JOIN DIM_Demographics d ON f.demographic_key = d.demographic_key
WHERE f.prevalence_per_1000 IS NOT NULL
  AND f.sample_size IS NOT NULL
"""

engine = create_engine(connection_string)
df = pd.read_sql(sql_query, engine)
engine.dispose()

print(f"Loaded {len(df)} studies")

# Preprocess data
print("\nPreprocessing data...")

df['male_female_ratio'] = df['male_female_ratio'].fillna(df['male_female_ratio'].median())
df['min_age'] = df['min_age'].fillna(0)
df['max_age'] = df['max_age'].fillna(18)
df['age_range'] = df['max_age'] - df['min_age']
df['is_nationwide'] = df['is_nationwide'].map({True: 1, False: 0, 'true': 1, 'false': 0})

label_encoders = {}
categorical_features = ['country', 'case_identification_method', 'case_criterion']

for col in categorical_features:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

print("Data preprocessed")

# Prepare features
print("\nPreparing features...")

feature_columns = [
    'year',
    'sample_size',
    'male_female_ratio',
    'is_nationwide',
    'min_age',
    'max_age',
    'age_range',
    'country_encoded',
    'case_identification_method_encoded',
    'case_criterion_encoded'
]

df_clean = df.dropna(subset=feature_columns + ['prevalence_per_1000'])

X = df_clean[feature_columns]
y = df_clean['prevalence_per_1000']

print(f"Features: {X.shape[1]}")
print(f"Samples: {X.shape[0]}")

# Split data
print("\nSplitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
print("\nTraining Random Forest model...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("Model trained")

# Make predictions
print("\nMaking predictions...")

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

print("Predictions complete")

# Evaluate model
print("\n" + "=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)

train_r2 = r2_score(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)

test_r2 = r2_score(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_mae = mean_absolute_error(y_test, y_test_pred)

print("\nTRAINING SET:")
print(f"   R² Score: {train_r2:.4f}")
print(f"   RMSE: {train_rmse:.4f}")
print(f"   MAE: {train_mae:.4f}")

print("\nTEST SET:")
print(f"   R² Score: {test_r2:.4f}")
print(f"   RMSE: {test_rmse:.4f}")
print(f"   MAE: {test_mae:.4f}")

print("\nINTERPRETATION:")
print(f"The model explains {test_r2*100:.1f}% of variance in prevalence rates")
print(f"Average prediction error: ±{test_mae:.2f} per 1,000 children")

# Feature importance
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"   {row['Feature']:40s} {row['Importance']:.4f}")

# Sample predictions
print("\nSample Predictions (First 10 Test Cases):")
print(f"{'Actual':<10} {'Predicted':<12} {'Error':<10}")
print("-" * 35)
for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_test_pred[i]
    error = actual - predicted
    print(f"{actual:<10.2f} {predicted:<12.2f} {error:<10.2f}")

print("\n" + "=" * 80)
print("MACHINE LEARNING COMPLETE")
print("=" * 80)
