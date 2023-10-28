Sales Predictor

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv("sales.csv")
print(data.head())
print("\n", data.isnull().sum())

figure = px.scatter(data_frame = data, x = "Sales", y = "TV", size = "TV", trendline = "ols")
figure.show()

figure = px.scatter(data_frame = data, x = "Sales", y = "Newspaper", size = "Newspaper", trendline = "ols")
figure.show()

figure = px.scatter(data_frame = data, x = "Sales", y = "Radio", size = "Radio", trendline = "ols")
figure.show()

correlation = data.corr()
print("\n", correlation["Sales"].sort_values(ascending = False))

x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)
print("\n", model.score(x_test, y_test))

features = np.array([[230.1, 37.8, 69.2]])
print("\n", model.predict(features))
