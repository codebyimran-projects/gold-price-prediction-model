import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset

dataset = pd.read_csv("gold.csv")

# Drop any null values
dataset = dataset.dropna()


# Features and target

# Features (all columns except Date and Close/Last)
X = dataset.drop(columns=["Date", "Close/Last"], axis=1)

# Target (gold price)
y = dataset["Close/Last"]


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3
)


# Train Linear Regression Model

model = LinearRegression()
model.fit(X_train, y_train)


# Predict on test set

y_pred = model.predict(X_test)


# Evaluate the model

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("R2 Score:", round(r2, 4))


# Predict single example

single_example = X.iloc[0]  # first row
single_example_reshaped = single_example.values.reshape(1, -1)
predicted_price = model.predict(single_example_reshaped)

print("\nSingle Example Prediction:")
print("Features:", single_example.to_dict())
print("Predicted Price:", round(predicted_price[0], 2))
