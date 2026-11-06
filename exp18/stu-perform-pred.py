import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset (UCI file uses semicolon separator)
DATA_PATH = Path(__file__).parent / "data" / "student-mat.csv"
df = pd.read_csv(DATA_PATH, sep=";")

print("\nFirst 5 rows:\n", df.head())
print("\nShape:", df.shape)
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe(include="all"))

# Remove duplicates and rows with missing target
df = df.drop_duplicates()
df = df.dropna(subset=["G3"])

# EDA
plt.figure(figsize=(7, 4))
sns.histplot(df["G3"], bins=15, kde=True)
plt.title("Final Grade Distribution")
plt.xlabel("Final Grade")
plt.savefig("student_grade_distribution.png")
plt.show()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.savefig("studytime_vs_grade.png")
plt.show()

# Separate features and target
X = df.drop(columns=["G3"])
y = df["G3"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Preprocessing
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train Random Forest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100, random_state=42
    ))
])

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)

print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", np.sqrt(mse))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model
joblib.dump(model, "student_performance_model.pkl")
print("\nModel saved successfully!")
