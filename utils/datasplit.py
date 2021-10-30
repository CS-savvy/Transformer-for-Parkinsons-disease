import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
import json

data_csv_file = Path.cwd().parent / "data/pd_speech_features.csv"
df = pd.read_csv(data_csv_file, skiprows=[0])
df_patient = df.drop_duplicates(subset='id')
df_patient.reset_index(inplace=True)
df_patient.drop(columns='index', inplace=True)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=450)

split_info = {}
cnt = 0
for train_indexes, test_indexes in skfold.split(df_patient, df_patient['class']):
    cnt+=1
    train_df = df_patient.iloc[train_indexes]
    test_df = df_patient.iloc[test_indexes]
    split_info[f'train_{cnt}'] = list(train_df['id'])
    split_info[f'val_{cnt}'] = list(test_df['id'])

with open(Path.cwd().parent / "data/split_details.json", 'w', encoding='utf8') as f:
    json.dump(split_info, f)
