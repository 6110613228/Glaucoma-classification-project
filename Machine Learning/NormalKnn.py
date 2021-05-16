# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_roc_curve
import joblib

#read in the data using pandas
df = pd.read_csv('trainingset.csv')




# %%
df = df.replace({'target':2},0)
df


# %%
kfold = 5
skf = StratifiedKFold(n_splits=kfold)
target = df.loc[:,'target']


# %%
from sklearn.model_selection import GridSearchCV


# %%
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

model = KNeighborsClassifier(n_neighbors= 56)
#param_grid = {'n_neighbors': np.arange(1, 100)}
#knn_gscv = GridSearchCV(model, param_grid, cv=5)
#knn_gscv.fit(X, y)
#knn_gscv.best_params_


# %%
def train_model(train, test, fold_no):
   X = ['CDR','DCD']
   y = ['target']
   X_train = train[X]
   y_train = train[y]
   X_test = test[X]
   y_test = test[y]
   model.fit(X_train,y_train)
   predictions = model.predict(X_test)
   filename = 'model/NormalkNN_fold_no'+str(fold_no)+'.sav'
   joblib.dump(model, filename)
   print('Fold',str(fold_no),
         'Accuracy:',
         accuracy_score(y_test,predictions))
   return X_test,y_test


# %%
fold_no = 1
trainset= []
testset = []
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
for train_index, test_index in skf.split(df, target):
    train = df.loc[train_index,:]
    trainset.append(train)
    test = df.loc[test_index,:]
    testset.append(test)
    X_test,y_test = train_model(train,test,fold_no)
    y = model.predict(X_test)
    
    
# Calculate Area under the curve to display on the plot
    viz = plot_roc_curve(model, X_test, y_test,
                         name='ROC fold {}'.format(fold_no),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    fold_no += 1
# Now, plot the computed values
    
    
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic(ROC):Normal and Non-Normal')
plt.legend(loc="lower right")
plt.show()   # Display


# %%


model = joblib.load('model/NormalkNN_fold_no1.sav')
conf_matrix = confusion_matrix(y_test,model.predict(X_test))
model.predict
FP = conf_matrix[0][1]
FN = conf_matrix[1][0]
TP = conf_matrix[1][1]
TN = conf_matrix[0][0]
Accuracy = (TP+TN)/(TP+FP+FN+TN)
Specificity = TN/(TN+FP)
sensitivity = TP / (TP + FN)
recall = TP + FP
F_score = 2* ((recall*sensitivity)/(recall+sensitivity))
print("Normal and Non-Normal")
print("Accuracy = ",Accuracy)
print("Specificity = ",Specificity)
print("sensitivity = ",sensitivity )
print("Fscore = ",F_score)
print(conf_matrix)


