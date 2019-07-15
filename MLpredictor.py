import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("/root/Desktop/jupyter/FINAL.CSV")

y = dataset.iloc[: , -1]
x = dataset.iloc[: , 0:5]

type = dataset.iloc[: ,0]
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
type_new = labelencoder_x.fit_transform(type)

x.iloc[: , 0] = type_new

from sklearn.preprocessing  import OneHotEncoder
onehotencoder = OneHotEncoder( categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

x= x[: , 1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train ,y_train)
model.predict(X_test)
print(y_test)
print(model.coef_)

