# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Avantika Krishnadas Kundooly
RegisterNumber: 212224040040
```

## Data head:
```py
import pandas as pd
data=pd.read_csv("Employee.csv")
display(data.head())
```

## Output:
<img width="1011" height="185" alt="image" src="https://github.com/user-attachments/assets/ac39e72f-170e-4203-958e-e358ccbf0917" />



## Dataset info:
```py
data.info()
```

## Output:
<img width="424" height="297" alt="image" src="https://github.com/user-attachments/assets/26d13a43-0213-4f5a-8c7a-9ff3c635e7ac" />



## Null dataset:
```py
display(data.isnull().sum())
```

## Output:
<img width="217" height="193" alt="image" src="https://github.com/user-attachments/assets/bb5fb54f-16ff-4c96-b504-b1bf10b68b29" />




## Values in left column:
```py
display(data['left'].value_counts())
```

## Output;
<img width="201" height="59" alt="image" src="https://github.com/user-attachments/assets/461e3a91-6833-4057-b725-4b6abf0246e8" />



## Prediction calculating code:
```py
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['salary']=le.fit_transform(data['salary'])
data.head()

x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','salary']]
x.head()

y=data['left']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_predict=dt.predict(x_test)
```

## Accuracy:
```py
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_predict)
print(accuracy)
```

## Output:
<img width="155" height="27" alt="image" src="https://github.com/user-attachments/assets/0e209a44-4805-4995-8f07-0ffa26921a11" />



## Prediction:
```py
print(dt.predict([[0.5,0.8,9,206,6,0,1,2]]))
```

## Output:
<img width="1005" height="95" alt="image" src="https://github.com/user-attachments/assets/b2772d7a-dbe1-4deb-863e-333454275417" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
