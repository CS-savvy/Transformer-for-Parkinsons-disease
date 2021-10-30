from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
from collections import Counter
import config
import json

this_dir = Path.cwd()
csv_file = this_dir / "data/pd_speech_features.csv"
df = pd.read_csv(csv_file, skiprows=[0])
df.drop(columns=['id'], inplace=True)
skip_column = ['gender', 'class']
columns = list(df.columns)
columns_for_norm = [c for c in columns if c not in skip_column]
for col in columns_for_norm:
    df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

columns = columns[:-1]
data = df.to_numpy(dtype=np.float32)

features, labels = data[:, :-1], data[:, -1]

oversample = ADASYN(random_state=config.SMOTE_SEED)

X_train, y_train = oversample.fit_resample(features, labels)
counter = Counter(y_train)
print(counter)

model = XGBClassifier(random_state=config.ML_SEED)
model.fit(X_train, y_train)
importance = model.feature_importances_
# summarize feature importance
feature_imp = []
for c, v in zip(columns, importance):
    feature_imp.append((c, float(v)))
    # print('Feature: %s, Score: %.5f' % (c, v))

feature_imp_sorted = sorted(feature_imp, key=lambda x: x[1], reverse=True)
with open("data/xgboost_feature_ranking.json", 'w', encoding='utf8') as f:
    json.dump(dict(feature_imp_sorted), f)

with open("data/xgboost_feature_ranking.json", 'r', encoding='utf8') as f:
    reloaded = json.load(f)

# print(reloaded[:32])

# final_str = ''
# for i, f in enumerate(reloaded[:32]):
#     i = i+1
#     fname = f[0]
#     id_name = '(F' + str(i) + ")"
#     final_str = final_str + fname + id_name + ", "
#
# print(final_str)
#
# # for i, v in enumerate(feature_imp_sorted):
#     print('Index: %d Feature: %s, Score: %.5f' % (i, v[0], v[1]))
#
# # plot feature importance
# pyplot.bar([x for x in range(len(importance))], [f[1] for f in feature_imp_sorted])
# pyplot.show()