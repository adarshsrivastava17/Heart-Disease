import numpy as np
import pandas as pd

data = pd.read_csv('C:/Users/adars/OneDrive/Desktop/Yukti-project/Heart-Disease/heart-disease.csv')
print(data)

data.head()

data.describe

data.info()

data.isnull().sum()

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
print(x)
print(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size= 0.2 , random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

print(x_train)
print(x_test)
print(y_train)
print(y_test)

from sklearn.ensemble import RandomForestClassifier
Classifier= RandomForestClassifier(n_estimators=100, random_state=42)
Classifier.fit(x_train, y_train)

y_pred = Classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.ensemble import AdaBoostClassifier
Adb = AdaBoostClassifier(n_estimators=100, algorithm="SAMME",random_state=0)
Adb.fit(x_train, y_train)

y_pred = Adb.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)
print(y_pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

result = Classifier.predict([[ 64,1,0,110,167,0,0,114]])
if result == 1:
  print("sorry to say but you are suffering with heart disease and you have to consult with Doctor !")
else:
  print("i am very happy to say that your are not suffereing from any kind of heart disease keep maintaing your regular diet .")