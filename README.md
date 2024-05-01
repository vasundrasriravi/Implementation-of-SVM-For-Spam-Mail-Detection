# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: VASUNDRA SRI R 
RegisterNumber: 212222230168 
```
```
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('/content/spam.csv',encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X=vectorizer.fit_transform(df['v2'])
y=df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model=svm.SVC(kernel='linear')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))
print("Classification Report:")
print(classification_report(y_test, predictions))
```
## Output:
## Head:
![Screenshot 2024-05-01 144954](https://github.com/vasundrasriravi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393983/31713086-b422-44b4-a2dd-370b06244cc1)


## Kernel Model:
![Screenshot 2024-05-01 145003](https://github.com/vasundrasriravi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393983/bfeb7692-525e-4afe-ba88-c123756d088f)


## Accuracy and Classification report:
![Screenshot 2024-05-01 145013](https://github.com/vasundrasriravi/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393983/ebd7a729-1c1e-4756-902e-8b12722a1a9e)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
