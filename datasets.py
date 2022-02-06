import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from imblearn.over_sampling import ADASYN
import warnings
import json
import yaml
warnings.filterwarnings("ignore")


class DatasetManager():
    def __init__(self):
        self.dataset_map = {'Parkinsion': ParkinsonsDataset, 'Gender': GenderDataset,
                            'Parkinsion-mx': ParkinsionMxDataset, 'Philippine': PhilippineDataset,
                            'Emotion': EmotionDataset}

    def get_dataset(self, name: str):
        return self.dataset_map[name]


class ParkinsonsDataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, indices: list, params: dict):
        """
        Args:
            csv_file (Path): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(Path(params['MainCSV']), skiprows=[0])
        self.feature_score_file = Path(params['FeatureImp'])
        self.max_features = params['NumFeatures'] if params['NumFeatures'] else self.feature_frame.shape[1] - 2
        self._filter_feature()
        self.feature_frame['valid'] = self.feature_frame.apply(lambda x: x['id'] in indices, axis=1)
        self.feature_frame = self.feature_frame[self.feature_frame['valid']]
        self.feature_frame = self.feature_frame.drop(columns=['id', 'valid'])
        self.feature_frame = self.feature_frame.reset_index(drop=True)
        self.feature_details = pd.read_csv(Path(params['FeatureDist']), index_col='Feature Type')
        self._preprocess()
        self.SMOTE = params['Smote']
        self.SMOTE_SEED = params['SmoteSeed']
        if self.SMOTE:
            self.numpy_data = self.feature_frame.to_numpy(dtype=np.float32)
            self.numpy_feature = self.numpy_data[:, :-1]
            self.numpy_label = self.numpy_data[:, -1]
            self._oversample()

    def _preprocess(self):
        df = self.feature_frame
        if 'id' in df:
            df.drop(columns=['id'], inplace=True)
        skip_column = ['index', 'gender', 'class']
        columns = list(df.columns)
        columns = [c for c in columns if c not in skip_column]
        for col in columns:
            mean = self.feature_details[self.feature_details['Features'] == col]['mean'][0]
            std = self.feature_details[self.feature_details['Features'] == col]['std'][0]
            df[col] = (df[col] - mean)/std
        self.feature_frame = df

    def _oversample(self):
        oversampler = ADASYN(random_state=self.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.numpy_feature, self.numpy_label)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            to_keep.append('class')
            to_keep = ['id'] + to_keep
            self.feature_frame = self.feature_frame[to_keep]

    def get_num_feature_length(self):
        return self.max_features

    def __len__(self):
        if self.SMOTE:
            return len(self.numpy_feature)
        return len(self.feature_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            return features, label

        datapoint = self.feature_frame.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        return features, label

    def collate(self, samples):
        features, label = map(list, zip(*samples))
        return torch.from_numpy(np.stack(features)), torch.Tensor(label)


class GenderDataset(Dataset):

    def __init__(self, indices: list, params: dict):
        self.df = pd.read_csv(Path(params['MainCSV']))
        self.df['class'] = self.df.apply(lambda x: 1 if x['label'] == 'male' else 0, axis=1)
        self.df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
        self.feature_score_file = Path(params['FeatureImp'])
        self.mapping_file = Path(params['FeatureDist'])
        self.num_features = params['NumFeatures']
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.normalize()
        self.SMOTE = params['Smote']
        self.SMOTE_SEED = params['SmoteSeed']
        if self.SMOTE:
            self.data = self.df.to_numpy(dtype=np.float32)
            self.features, self.labels = self.data[:, :-1], self.data[:, -1]
            self._oversample()
        return

    def normalize(self):
        with open(self.mapping_file, 'r', encoding='utf8') as f:
            mapping = json.load(f)
        columns = list(self.df.columns)
        columns = [c for c in columns if c not in self.skip_column]
        for col in columns:
            self.df[col] = (self.df[col] - mapping[col]['mean']) / mapping[col]['std']
        return

    def _oversample(self):
        oversampler = ADASYN(random_state=self.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.features, self.labels)

    def get_data(self, indices):
        X = self.features[indices]
        y = self.labels[indices]
        return X, y

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.num_features]]
            to_keep.append('class')
            self.df = self.df[to_keep]

    def __len__(self):
        if self.SMOTE:
            return len(self.numpy_feature)
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            return features, label

        datapoint = self.df.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        return features, label

    def collate(self, samples):
        features, label = map(list, zip(*samples))
        return torch.from_numpy(np.stack(features)), torch.Tensor(label)


class ParkinsionMxDataset(Dataset):
    """parkinson's multiple phonation dataset."""

    def __init__(self, indices: list, params: dict):
        """
        Args:
            csv_file (Path): Path to the csv file with features and label.
        """
        self.feature_frame = pd.read_csv(Path(params['MainCSV']))
        if 'UPDRS' in self.feature_frame:
            self.feature_frame.drop(columns=['UPDRS'], inplace=True)
        self.feature_score_file = Path(params['FeatureImp'])
        self.max_features = params['NumFeatures'] if params['NumFeatures'] else self.feature_frame.shape[1] - 2 # minus 2 for id and class
        self._filter_feature()
        self.feature_frame['valid'] = self.feature_frame.apply(lambda x: x['Subject id'] in indices, axis=1)
        self.feature_frame = self.feature_frame[self.feature_frame['valid']]
        self.feature_frame = self.feature_frame.drop(columns=['Subject id', 'valid'])
        self.feature_frame = self.feature_frame.reset_index(drop=True)
        self.feature_details = Path(params['FeatureDist'])
        self.skip_column = ['class']
        self.normalize()
        self.SMOTE = params['Smote']
        self.SMOTE_SEED = params['SmoteSeed']
        if self.SMOTE:
            self.numpy_data = self.feature_frame.to_numpy(dtype=np.float32)
            self.numpy_feature = self.numpy_data[:, :-1]
            self.numpy_label = self.numpy_data[:, -1]
            self._oversample()

    def normalize(self):
        with open(self.feature_details, 'r', encoding='utf8') as f:
            mapping = json.load(f)
        columns = list(self.feature_frame.columns)
        columns = [c for c in columns if c not in self.skip_column]
        for col in columns:
            self.feature_frame[col] = (self.feature_frame[col] - mapping[col]['mean']) / mapping[col]['std']
        return

    def _oversample(self):
        oversampler = ADASYN(random_state=self.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.numpy_feature, self.numpy_label)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            to_keep.append('class')
            to_keep = ['Subject id'] + to_keep
            self.feature_frame = self.feature_frame[to_keep]

    def __len__(self):
        if self.SMOTE:
            return len(self.numpy_feature)
        return len(self.feature_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.SMOTE:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            return features, label
        datapoint = self.feature_frame.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        return features, label

    def collate(self, samples):
        features, label = map(list, zip(*samples))
        return torch.from_numpy(np.stack(features)), torch.Tensor(label)


class PhilippineDataset(Dataset):

    def __init__(self, indices: list, params: dict):
        self.df = pd.read_csv(Path(params['MainCSV']))
        self.feature_score_file = Path(params['FeatureImp'])
        self.feature_details = Path(params['FeatureDist'])
        self.max_features = params['NumFeatures']
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.normalize()

    def normalize(self):
        with open(self.feature_details, 'r', encoding='utf8') as f:
            mapping = json.load(f)
        columns = list(self.df.columns)
        columns = [c for c in columns if c not in self.skip_column]
        for col in columns:
            self.df[col] = (self.df[col] - mapping[col]['mean']) / mapping[col]['std']
        return

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            to_keep.append('class')
            self.df = self.df[to_keep]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.df.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        return features, label

    def collate(self, samples):
        features, label = map(list, zip(*samples))
        return torch.from_numpy(np.stack(features)), torch.Tensor(label)


class EmotionDataset(Dataset):

    def __init__(self, indices: list, params: dict):
        self.df = pd.read_csv(Path(params['MainCSV']))
        self.cmap = ['happy', 'surprised', 'angry']
        self.df['class'] = self.df.apply(lambda x: self.cmap.index(x['Emotion ']), axis=1)
        self.df.drop(columns=['name', 'Emotion ', 'Type'], inplace=True)
        self.feature_score_file = Path(params['FeatureImp'])
        self.num_features = params['NumFeatures']
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.SMOTE = params['Smote']
        self.SMOTE_SEED = params['SmoteSeed']
        if self.SMOTE:
            self.data = self.df.to_numpy(dtype=np.float32)
            self.features, self.labels = self.data[:, :-1], self.data[:, -1]
            self._oversample()

    def _oversample(self):
        self.oversampler = ADASYN(random_state=self.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = self.oversampler.fit_resample(self.features, self.labels)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.num_features]]
            to_keep.append('class')
            self.df = self.df[to_keep]

    def __len__(self):
        if self.SMOTE:
            return len(self.numpy_feature)
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            return features, label

        datapoint = self.df.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        return features, label

    def collate(self, samples):
        features, label = map(list, zip(*samples))
        return torch.from_numpy(np.stack(features)), torch.Tensor(label)


if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config['ExpDetails']['Dataset']
    data_split_file = config['Datasets'][dataset_name]['Split']
    with open(data_split_file, 'r', encoding='utf8') as f:
        split_detail = json.load(f)
    print("Dataset details: ", config['Datasets'][dataset_name])
    dm = DatasetManager()
    selected_dataset = dm.selector(dataset_name)
    dataset = selected_dataset(split_detail['train_1'], config['Datasets'][dataset_name])
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=dataset.collate)
    for i_batch, (batched_features, batched_target) in enumerate(train_dataloader):
        print(i_batch, batched_features.size(), batched_target.size())
