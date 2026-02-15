import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 
df=pd.read_csv("C:\\Users\\misht\\OneDrive\\Desktop\\ishu ml\\Titanic-Dataset.csv")
df['Sex']=df['Sex'].map({'male':0,'female':1})
df['Age']=df['Age'].fillna(df['Age'].mean())
x=df[["Pclass","Sex","Age"]]
y=df[["Survived"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LogisticRegression()
model.fit(x_train,y_train)
pclass=int(input("Enter Pclass:"))
Sex=str(input("Enter Sex:"))
if(Sex.lower()=="male" ):
    Sex_value=0
elif(Sex.lower()=="female"):
    Sex_value=1

age=int(input("Enter age:"))
prediction=model.predict([[pclass,Sex_value,age]])
if(prediction[0]==1):
    print("Prediction=Survived")
else:
    print("Prediction=Dead")

# print(f"{prediction:.2f}")
print(f"{model.score(x_test,y_test):.2f}")