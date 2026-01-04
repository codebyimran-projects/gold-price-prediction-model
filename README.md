
# Gold Price Prediction

Predict the price of gold using historical data with machine learning in Python.

## 📈 Project Overview
This project uses a **Linear Regression** model to predict the daily closing price of gold. Historical gold price data including Open, High, Low, and Volume is used as features to train the model.  

The model allows making predictions for future prices based on past trends.

---

## ⚙️ Features
- Data preprocessing and cleaning
- Handle missing values
- Train/Test split for model evaluation
- Predict gold price using Linear Regression
- Evaluate predictions with metrics like MAE, MSE, and R² score
- Single example prediction support

---

## 🛠️ Technologies
- Python 3
- Pandas
- NumPy
- scikit-learn (LinearRegression, train_test_split)

---

## 💻 Usage

1. Clone the repository:
```bash
git clone https://github.com/codebyimran-projects/gold-price-prediction.git
````

2. Navigate to project folder:

```bash
cd gold-price-prediction
```

3. Install dependencies:

```bash
pip install pandas numpy scikit-learn
```

4. Run the main script:

```bash
python main.py
```

5. Example prediction:

```python
# Single data example
single_example = X.iloc[0]
predicted_price = model.predict([single_example])
print(predicted_price)
```

---

## 📊 Model Evaluation

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

---

## 📁 Dataset

The dataset (`gold.csv`) includes:

* Date
* Open
* High
* Low
* Volume
* Close/Last (Target)

---

## 📌 GitHub

[Gold Price Prediction Repository](https://github.com/codebyimran-projects/gold-price-prediction)

---

## 🏷️ Hashtags

#Python #MachineLearning #DataScience #GoldPricePrediction #LinearRegression #MLProject #AI #StockMarket #Finance #Prediction

