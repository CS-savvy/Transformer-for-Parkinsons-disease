import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold, StratifiedKFold
import json

data_csv_file = Path("data/pd_speech_features.csv")
df = pd.read_csv(data_csv_file, skiprows=[0])
non_parkinson = df[df['class'] == 0]
parkinson = df[df['class'] == 1]
sample_parkinson = parkinson.sample(50)
sample_non_parkinson = non_parkinson.sample(50)
test_set_index = list(sample_parkinson.index) + list(sample_non_parkinson.index)
remainder_df = df.drop(index=test_set_index)
# kfold = KFold(n_splits=10, shuffle=True, random_state=450)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=450)
remainder_df.reset_index(inplace=True)

split_info = {}

cnt = 0
for train_indexes, test_indexes in skfold.split(remainder_df, remainder_df['class']):
    cnt+=1
    train_df = remainder_df.iloc[train_indexes]
    test_df = remainder_df.iloc[test_indexes]
    split_info[f'train_{cnt}'] = list(train_df['index'])
    split_info[f'val_{cnt}'] = list(test_df['index'])

split_info['test'] = test_set_index

with open("data/split_details.json", 'w', encoding='utf8') as f:
    json.dump(split_info, f)


# print("Non-parkinson",train_df[train_df['class'] == 0].shape[0] , test_df[test_df['class'] == 0].shape[0])
# print("parkinson", train_df[train_df['class'] == 1].shape[0], test_df[test_df['class'] == 1].shape[0])
