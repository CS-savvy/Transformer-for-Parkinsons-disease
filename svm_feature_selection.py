from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import KFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import numpy as np
import pandas as pd
from pathlib import Path

this_dir = Path.cwd()
csv_file = this_dir / "data/pd_speech_features.csv"
df = pd.read_csv(csv_file, skiprows=[0])
df.drop(columns=['id'], inplace=True)
skip_column = ['gender', 'class']
columns =list(df.columns)
columns = [c for c in columns if c not in skip_column]
for col in columns:
    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

data = df.to_numpy(dtype=np.float32)

feature_names = ['gender'] + columns
features, labels = data[:, :-1], data[:, -1]

sfs = SFS(svm.SVC(), k_features=256, forward=True, floating=False, scoring='accuracy', cv=0, n_jobs=16, verbose=1)
sfs_his = sfs.fit(features, labels, custom_feature_names=feature_names)
selected_features= sfs_his.k_feature_names_
print("Feature selected :", selected_features)
feature_subset = features[:, sfs_his.k_feature_idx_]

kfold = KFold(n_splits=10, shuffle=True, random_state=450)

# feature_subset = features

accuracy = []
precision = []
recall = []
for train_indexes, test_indexes in kfold.split(data):
    X_train, y_train = feature_subset[train_indexes], labels[train_indexes]
    X_test, y_test = feature_subset[test_indexes], labels[test_indexes]

    clf = svm.SVC(kernel='rbf', gamma=0.001, degree=1, C=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)

print("Avg accuracy:", sum(accuracy)/len(accuracy))
print("Avg precision:", sum(precision)/len(precision))
print("Avg recall:", sum(recall)/len(recall))