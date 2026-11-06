import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load dataset
DATA_PATH = Path(__file__).parent / "data" / "housePricePredictionDataset.csv"
df = pd.read_csv(DATA_PATH)

# Change this to the exact target column name in your CSV
TARGET = "Price"

print("\nFirst 5 rows:\n", df.head())
print("\nDataset shape:", df.shape)
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nStatistical summary:\n", df.describe(include="all"))

# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows where the target is missing
df = df.dropna(subset=[TARGET])

# EDA
plt.figure(figsize=(7, 4))
sns.histplot(df[TARGET], kde=True)
plt.title("House Price Distribution")
plt.savefig("house_price_distribution.png")
plt.show()

# Separate features and target
X = df.drop(columns=[TARGET])
y = df[TARGET]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = X.select_dtypes(exclude=np.number).columns

# Preprocess numerical and categorical features separately
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# Split before fitting preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "house_price_model.pkl")
print("\nModel saved successfully!")
