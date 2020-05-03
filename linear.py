import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle
from matplotlib import  style

data= pd.read_csv("student-mat.csv",sep=";")

#print(data.head())
#print(data.tail())

data= data[["G1","G2","G3","failures","absences"]]
predict="G3"

x=np.array(data.drop([predict],1))
y= np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

"""best=0

for _ in range(30):
     x_train,x_test,y_train,y_test= sklearn.model_selection.train_test_split(x,y,test_size=0.2)
     linear=linear_model.LinearRegression()
     linear.fit(x_train,y_train)
     acc= linear.score(x_test,y_test)
     print(acc)
     if acc> best:
         best=acc
         with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear,f)"""

pickle_in= open("studentmodel.pickle", "rb")

linear= pickle.load(pickle_in)
print("coeff: \n", linear.coef_)
print("Inter: \n", linear.intercept_)

prediction= linear.predict(x_test)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': prediction.flatten()})
#print(df)

for x in range(len(prediction)):
    print(prediction[x], x_test[x], y_test[x])

p="absences"
style.use('ggplot')
plt.scatter(data[p], data["G3"])
plt.xlabel(p)
plt.ylabel('Final grade')
plt.show()


"""plt.scatter(x_test[:,1], y_test,  color='gray')
plt.scatter(x_test[:,1], prediction, color='red', linewidth=1)
plt.show()"""