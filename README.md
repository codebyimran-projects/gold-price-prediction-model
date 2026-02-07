

# Gold Price Prediction Model

A machine learning project that predicts gold prices using historical market data and a Random Forest Regressor model built in Python.

---

## Project Overview

This project uses historical gold price data to train a machine learning model that predicts the **GLD (Gold ETF) price**.
The model learns patterns from related financial indicators such as oil prices, silver prices, and currency indexes.

The goal is to understand how different market factors influence gold prices and build an accurate predictive model.

---

## Dataset

* File: `gld_price_data.csv`
* Target column: `GLD`
* Dropped column: `Date` (non-numeric)

### Dataset Features

The dataset includes financial indicators such as:

* SPX (S&P 500 Index)
* USO (Oil ETF)
* SLV (Silver ETF)
* EUR/USD exchange rate
* GLD (Gold price – target)

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Machine Learning Model

* Algorithm: Random Forest Regressor
* Train/Test Split: 80% / 20%
* Evaluation Metric: R² Score

Random Forest was chosen because it handles non-linear relationships well and performs strongly on financial datasets.

---

## Workflow

1. Load and inspect dataset
2. Handle missing values
3. Drop non-numeric columns
4. Perform correlation analysis
5. Split data into training and testing sets
6. Train Random Forest Regressor
7. Evaluate model performance
8. Visualize actual vs predicted prices

---

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/codebyimran-projects/gold-price-prediction.git
```

### 2. Navigate to project directory

```bash
cd gold-price-prediction
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 4. Run the model

```bash
python main.py
```

---

## Model Evaluation

The model performance is evaluated using the **R² Score**, which measures how well the predicted values match the actual gold prices.

A higher R² score indicates better prediction accuracy.

---

## Visualization

The project includes a line plot comparing:

* Actual gold prices
* Predicted gold prices

This helps visually evaluate how well the model performs.

---

## Future Improvements

* Add MAE and RMSE metrics
* Feature importance visualization
* Hyperparameter tuning
* Save trained model using joblib
* Build a prediction API

---

## Author

**Developed by:** codebyimran
