from pathlib import Path
import pandas as pd

this_dir = Path.cwd()
dataset_file = this_dir / "data/pd_speech_features.ods"

df = pd.read_excel(dataset_file, header=[0, 1], engine="odf")
df_main = pd.read_csv("data/pd_speech_features.csv", skiprows=[0])
print(df.head())

feature_df = pd.DataFrame(list(df.columns.values), columns=['Feature Type', 'Features'])
print(feature_df.head())

stds = []
means = []
for i, ds in feature_df.iterrows():
    feature_name = ds['Features']
    means.append(df_main[feature_name].mean())
    stds.append(df_main[feature_name].std(ddof=0))

feature_df['mean'] = means
feature_df['std'] = stds

feature_df.to_csv("data/feature_details.csv", index=False)