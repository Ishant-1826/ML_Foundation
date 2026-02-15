import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
data = {
    'Calories': [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000],
    'Sleep': [7, 8, 6, 5, 8, 7, 6, 8],
    'WeightLoss': [1.2, 1.5, 2.0, 2.2, 2.8, 3.2, 3.5, 4.0],
    'IsHealthy': [1, 1, 0, 0, 1, 1, 0, 1]
}
df = pd.DataFrame(data)
x1=df[['Calories','Sleep']]
y1=df[['WeightLoss']]
x_train1,x_test1,y_train1,y_test1=train_test_split(x1,y1,test_size=0.2,random_state=42)
model1=LinearRegression()
model1.fit(x_train1,y_train1.values.ravel())
m1=model1.coef_[0]
m2=model1.coef_[1]
c=model1.intercept_
cal=int(input("Enter Calories:"))
sleep=int(input("Enter Hours of sleep:"))
input1=pd.DataFrame([[cal, sleep]], columns=['Calories', 'Sleep'])
y_pred1=model1.predict(input1)
print("The predicted WeightLoss =", y_pred1[0])
print(f"Model1 Accuracy= {model1.score(x_test1,y_test1)*100:.2f}%")
print(f"The LinearRegression Model equation: y={m1:.6f}*Calories+{m2:.6f}*Sleep+{c:.2f}")
x2=df[['Calories','WeightLoss']]
y2=df[['IsHealthy']]
x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.2,random_state=42)
model2=LogisticRegression()
model2.fit(x_train2,y_train2.values.ravel())
cal=int(input("Enter Calories:"))
wl=float(input("Enter Weightloss:"))
input2=pd.DataFrame([[cal, wl]], columns=['Calories', 'WeightLoss'])
y_pred2=model2.predict(input2)
if(y_pred2[0]==1):
    pred=("The person is Healthy")
elif(y_pred2[0]!=1):
    pred=("The person is Not Healthy")

print(f"The prediction is: {pred}")
print(f"Model2 Accuracy= {model2.score(x_test2,y_test2)*100:.2f}%")