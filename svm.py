from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
from pathlib import Path

this_dir = Path.cwd()
csv_file= this_dir / "data/pd_speech_features.csv"
df = pd.read_csv(csv_file, skiprows=[0])
df.drop(columns=['id'], inplace=True)
skip_column = ['gender', 'class']
columns =list(df.columns)
columns = [c for c in columns if c not in skip_column]
for col in columns:
    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

data = df.to_numpy(dtype=np.float32)

features, labels = data[:, :-1], data[:, -1]

param_grid = { 'C':[0.1,1,100,1000],
              'kernel':['rbf','poly','sigmoid','linear'],
              'degree':[1,2,3,4,5,6],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

kfold = KFold(n_splits=10, shuffle=True, random_state=450)

accuracy = []
precision = []
recall = []
for train_indexes, test_indexes in kfold.split(data):
    X_train, y_train = features[train_indexes], labels[train_indexes]
    X_test, y_test = features[test_indexes], labels[test_indexes]
    clf = svm.SVC()
    grid = GridSearchCV(clf, param_grid, n_jobs=12, cv=5, scoring='accuracy', verbose=1)
    grid.fit(X_train, y_train)
    print(grid.best_params_)

    y_pred = grid.predict(X_test)

    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))

print("Avg accuracy:", sum(accuracy)/len(accuracy))
print("Avg precision:", sum(precision)/len(precision))
print("Avg recall:", sum(recall)/len(recall))