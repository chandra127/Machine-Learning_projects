import numpy as np
import sklearn
from sklearn import preprocessing,model_selection
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd

df= pd.read_csv("breast-cancer-wisconsin.data")

df.replace('?',-99999, inplace=True)
df.drop(['id'],1,inplace=True)
#print(df.head())

X=np.array(df.drop(['class'],1))
Y=np.array(df['class'])

X_train,X_test,Y_train,Y_test=sklearn.model_selection.train_test_split(X,Y,test_size=0.2)

model = KNeighborsClassifier()
model.fit(X_train, Y_train)
accuracy = model.score(X_test, Y_test)

print(accuracy)

#if we want to add new test data
patient_measures= np.array([[4,2,1,1,2,2,3,2,1],[4,2,2,2,2,2,3,2,1]])
patient_measures=patient_measures.reshape(len(patient_measures),-1)

#predicted= model.predict(X_test)
predicted= model.predict(patient_measures)

print(predicted)