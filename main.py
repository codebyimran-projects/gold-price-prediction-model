import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# load csv file 
gld_price_data = pd.read_csv('gld_price_data.csv')
print(gld_price_data.head())

# check dataset lenght 
print(gld_price_data.shape)
# check null values 
print(gld_price_data.isnull().sum())
gld_price_data = gld_price_data.drop('Date', axis=1)
correlation = gld_price_data.corr()

# plt.figure(figsize=(8, 8))
# sns.heatmap(
#     correlation,
#     annot=True,
#     fmt='.1f',
#     cmap='Blues',
#     square=True,
#     cbar=True,
#     annot_kws={'size': 8}
# )

# plt.show()

# print(correlation['GLD'])

# splitting data into two var 
X = gld_price_data.drop('GLD', axis=1)
y = gld_price_data['GLD']

print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2
)

print (X_train.shape)



model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred_data = model.predict(X_test)

r2_error = metrics.r2_score(y_test, pred_data)

print("R square error:", r2_error)