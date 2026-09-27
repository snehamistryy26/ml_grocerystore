import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_FILE = "Grocery_Sales_Retail_Analytics_24bcads054.csv"
MODEL_FILE = "train_model.pkl"
PREDICTIONS_FILE = "test_predictions.csv"


# Load dataset
df = pd.read_csv(DATA_FILE)

print("Dataset loaded successfully!")
print("Dataset Shape:", df.shape)


# Convert Order Date
df["Order Date"] = pd.to_datetime(
    df["Order Date"].astype(str),
    dayfirst=True,
    format="mixed",
    errors="coerce"
)

# Fill invalid dates
median_date = df["Order Date"].dropna().median()
df["Order Date"] = df["Order Date"].fillna(median_date)


# Create date features
df["Order_Year"] = df["Order Date"].dt.year
df["Order_Month"] = df["Order Date"].dt.month
df["Order_Day"] = df["Order Date"].dt.day


# Target
TARGET = "Profit"


# Features
FEATURES = [
    "Category",
    "Sub Category",
    "City",
    "Region",
    "State",
    "Sales",
    "Discount",
    "Order_Year",
    "Order_Month",
    "Order_Day"
]


categorical_features = [
    "Category",
    "Sub Category",
    "City",
    "Region",
    "State"
]


numeric_features = [
    "Sales",
    "Discount",
    "Order_Year",
    "Order_Month",
    "Order_Day"
]


X = df[FEATURES]
y = pd.to_numeric(df[TARGET], errors="coerce")


# Remove missing target values
valid_rows = y.notna()

X = X.loc[valid_rows]
y = y.loc[valid_rows]


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]),
            categorical_features
        ),

        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_features
        )
    ]
)


# Machine Learning Model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# Complete Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# Train model
print("\nTraining model...")

pipeline.fit(X_train, y_train)


# Prediction
y_pred = pipeline.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)


print("\nModel Training Completed")
print("--------------------------------")
print("MAE      :", round(mae, 2))
print("RMSE     :", round(rmse, 2))
print("R2 Score :", round(r2, 4))


# Save model
joblib.dump(
    pipeline,
    MODEL_FILE
)

print("\nModel saved as:", MODEL_FILE)


# Save predictions
predictions = X_test.copy()

predictions["Actual Profit"] = y_test.values

predictions["Predicted Profit"] = np.round(
    y_pred,
    2
)

predictions.to_csv(
    PREDICTIONS_FILE,
    index=False
)

print(
    "Predictions saved as:",
    PREDICTIONS_FILE
)