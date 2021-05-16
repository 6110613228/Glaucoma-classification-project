import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

#read in the data using pandas
df = pd.read_csv('newTrain.csv')

# drop coumns target and index
X = df.drop(columns=['target','index'])

#separate target values
y = df['target'].values

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5, stratify=y)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 83)

# Fit the classifier to the data
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)

#check accuracy of our model on the test data
knn.score(X_test, y_test)

#train model with cv of 5 
cv_scores = cross_val_score(knn, X, y, cv=5)

#print each cv score (accuracy) and average them
print(cv_scores)
print('cv_scores mean:{}'.format(np.mean(cv_scores)))

#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 100)}

#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv=5)

#fit model to data
knn_gscv.fit(X, y)

#check top performing n_neighbors value
print("best n_neighbors : ",knn_gscv.best_params_)
joblib.dump(knn, "./model/KNN.sav")

# test with testset
loaded_model = joblib.load("./model/KNN.sav")
loaded_model.predict(X_test)
dfTest = pd.read_csv('newTest.csv')
X_test = dfTest.drop(columns=['target','index'])
y_test = dfTest['target'].values
y_pred = loaded_model.predict(X_test)

# show accuracy
print("acc : ",metrics.accuracy_score(y_test, y_pred))