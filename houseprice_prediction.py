import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv("C:\\Users\\misht\\OneDrive\\Desktop\\ishu ml\\Housing.csv")
x = df[["area"]]
y=df[["price"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train.values.ravel())
m=model.coef_[0]#get coefficient of x that is m=slope
c=model.intercept_
print(f"Scikit-Learn Model: y = {m:.2f}x + {c:.2f}")
print(f"Model Score ={model.score(x_test,y_test)}")
a = pd.DataFrame([[3000]], columns=["area"])
prediction=model.predict(a)
print(f"{prediction[0]:.2f}")
plt.scatter(x,y)
plt.show()