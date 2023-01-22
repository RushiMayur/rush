import numpy as np
import pandas as pd


from sklearn.metrics import accuracy_score

# Train Dataset
data = pd.read_csv('traininsta.csv')



##Train Dataset
# data_train=pd.read_csv('traininsta.csv')
#data_train.fillna(data_train.mean(), inplace=True)

# Test Dataset
# data_test=pd.read_csv('testinsta.csv')
#data_test.fillna(data_test, inplace=True)

# Concatinate test and train dataset

df = data.fillna(0)
#print(data.isnull().values.any())
#data.fillna(data, inplace=True)

# Value counts

Z = data.drop(columns=['fake'])
Y = data['fake']

print(Z.head(5))



# Test-Train split
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(Z, Y, test_size=0.1, random_state=7)

#print(X_train.isnull().values.any())
#print(X_test.isnull().values.any())
#print(Y_test.isnull().values.any())
#print(Y_train.isnull().values.any())

# Logistic Regression

from sklearn.ensemble import HistGradientBoostingClassifier

rf = HistGradientBoostingClassifier()
rf.fit(X_train, Y_train)
pred_1 = rf.predict(X_test)
print(accuracy_score(Y_test,pred_1))

import pickle

pickle.dump(rf, open('model.pkl', 'wb'))
