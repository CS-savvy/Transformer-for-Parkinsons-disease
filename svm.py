from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib


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

with open("data/split_details.json", 'r', encoding='utf8') as f:
    split_detail = json.load(f)

k_fold = 10

accuracy = []
precision = []
recall = []

model_dir = Path.cwd() / "models/SVM/"
if not model_dir.exists():
    model_dir.mkdir(parents=True)

for i in range(1, k_fold + 1):
    X_train, y_train = features[split_detail[f'train_{i}']], labels[split_detail[f'train_{i}']]
    X_test, y_test = features[split_detail[f'val_{i}']], labels[split_detail[f'val_{i}']]
    # clf = svm.SVC()
    # grid = GridSearchCV(clf, param_grid, n_jobs=12, cv=5, scoring='accuracy', verbose=1)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)
    # y_pred = grid.predict(X_test)

    clf = svm.SVC(kernel='rbf', gamma=0.001, degree=1, C=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    _ = joblib.dump(clf, model_dir / f"model_k_fold_{i}.pkl", compress=0)

print("Avg accuracy:", sum(accuracy)/len(accuracy))
print("Avg precision:", sum(precision)/len(precision))
print("Avg recall:", sum(recall)/len(recall))