import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Load dataset
gld_price_data = pd.read_csv('gld_price_data.csv')

# Basic checks
print("Dataset shape:", gld_price_data.shape)
print("Missing values:\n", gld_price_data.isnull().sum())

# Drop Date column (non-numeric)
gld_price_data = gld_price_data.drop('Date', axis=1)

# Correlation analysis
correlation = gld_price_data.corr()

# Heatmap (optional)
# plt.figure(figsize=(8, 8))
# sns.heatmap(
#     correlation,
#     annot=True,
#     fmt='.1f',
#     cmap='Blues',
#     square=True,
#     cbar=True
# )
# plt.show()

# Split features and target
X = gld_price_data.drop('GLD', axis=1)
y = gld_price_data['GLD']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

# Model initialization
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
pred_data = model.predict(X_test)

# Evaluation
r2_error = metrics.r2_score(y_test, pred_data)
print("R² Score:", r2_error)

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label='Actual Values', color='blue')
plt.plot(pred_data, label='Predicted Values', color='green')
plt.title('Actual vs Predicted Gold Prices')
plt.xlabel('Sample Index')
plt.ylabel('Gold Price')
plt.legend()
plt.tight_layout()
plt.show()
