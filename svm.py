from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

import numpy as np
import pandas as pd
from pathlib import Path

this_dir = Path.cwd()

dataset_file = this_dir / "data/pd_speech_features.ods"
df = pd.read_excel(dataset_file, header=[0, 1], engine="odf")

print("Dataset Loaded !!")
df.drop(columns=['Basic Info'], inplace=True)

data = df.to_numpy(dtype=np.float32)

features, labels = data[:, :-1], data[:, -1]

scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=450) # 70% training and 30% test

clf = svm.SVC(kernel='rbf', verbose=1) # Linear Kernel
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))