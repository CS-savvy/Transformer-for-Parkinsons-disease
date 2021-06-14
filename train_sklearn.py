from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from collections import Counter
from sklearn.decomposition import PCA
import os


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

# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }

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

    # SMOTE
    # oversample = SMOTE()
    # oversample =BorderlineSMOTE()
    oversample = ADASYN()

    X_train, y_train = oversample.fit_resample(X_train, y_train)
    counter = Counter(y_train)
    print(counter)

    ## PCA
    # pca = PCA(n_components=590)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # print(X_train.shape)
    # X_test = pca.transform(X_test)

    ## Classifier
    # clf = svm.SVC(probability=True)
    # clf = RandomForestClassifier()
    # clf = AdaBoostClassifier()
    # clf = GradientBoostingClassifier()
    clf = XGBClassifier()
    # clf = MLPClassifier()
    # clf = KNeighborsClassifier()
    # clf = DecisionTreeClassifier()
    # cnlf = LogisticRegression(max_iter=1000)
    # clf = GaussianNB()

    ## Grid search
    # grid = GridSearchCV(clf, param_grid, n_jobs=12, cv=5, scoring='accuracy', verbose=1)
    # grid.fit(X_train, y_train)
    # print(grid.best_params_)
    # y_pred = grid.predict(X_test)


    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    _ = joblib.dump(clf, model_dir / f"model_k_fold_{i}.pkl", compress=0)

print("Avg accuracy:", sum(accuracy)/len(accuracy))
print("Avg precision:", sum(precision)/len(precision))
print("Avg recall:", sum(recall)/len(recall))

os.system('python eval_sklearn.py')

# def evaluate(model_dir, data, split_details=None):
#     conf_threshold = 0.5
#     features, labels = data
#     max_val_auc = 0
#     best_model = None
#     results = []
#     val_aucs = []
#     for i in range(1, 11):
#         val_indices = split_details[f'val_{i}']
#         X_val, y_val = features[val_indices], labels[val_indices]
#         X_val = pca.transform(X_val)
#         model_path = model_dir / f"model_k_fold_{i}.pkl"
#         clf = joblib.load(model_path)
#         y_score = clf.predict_proba(X_val)[:, 1]
#         y_pred = (y_score > conf_threshold)*1.0
#         accuracy = metrics.accuracy_score(y_val, y_pred)
#         precision = metrics.precision_score(y_val, y_pred)
#         recall = metrics.recall_score(y_val, y_pred)
#         roc_auc = metrics.roc_auc_score(y_val, y_score)
#         val_aucs.append(roc_auc)
#         print(f"Accuracy : {accuracy} | precision : {precision} | Recall : {recall} | ROC-AUC : {roc_auc}")
#         if roc_auc > max_val_auc:
#             max_val_auc = roc_auc
#             best_model = i
#         results.extend([(u, l, p, s, l == p) for u, l, p, s in zip(val_indices, y_val, y_pred, y_score)])
#     avg_val_auc = sum(val_aucs) / len(val_aucs)
#     print("Average val AUC :", avg_val_auc)
#     print(f"Best Model : {best_model} with AUC {max_val_auc}")
#     test_indices = split_details['test']
#     X_test, y_test = features[test_indices], labels[test_indices]
#     X_test = pca.transform(X_test)
#     model_path = model_dir / f"model_k_fold_{best_model}.pkl"
#     clf = joblib.load(model_path)
#     y_score = clf.predict_proba(X_test)[:, 1]
#     y_pred = (y_score > conf_threshold) * 1.0
#     accuracy = metrics.accuracy_score(y_test, y_pred)
#     precision = metrics.precision_score(y_test, y_pred)
#     recall = metrics.recall_score(y_test, y_pred)
#     roc_auc = metrics.roc_auc_score(y_test, y_score)
#     print(f"Test - Accuracy : {accuracy} | precision : {precision} | Recall : {recall} | ROC-AUC : {roc_auc}")
#     results.extend([(u, l, p, s, l==p) for u, l, p, s in zip(test_indices, y_test, y_pred, y_score)])
#
#     results = sorted(results, key=lambda x: x[0])
#     result_df = pd.DataFrame(results, columns=["UID", 'True Label', 'Prediction', 'Score', 'Match'])
#     result_df.to_csv(model_dir / f"eval_result.csv", index=False)
#
# evaluate(model_dir, (features, labels), split_details=split_detail)