import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import warnings
import json
import yaml
warnings.filterwarnings("ignore")


class ParkinsonsDataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, split_indices: list, params: dict):
        """
        Args:
            csv_file (Path): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(csv_file, skiprows=[0])
        self.feature_score_file = feature_score_file
        self.max_features = max_features if max_features else self.feature_frame.shape[1] - 2
        self._filter_feature()
        self.feature_frame['valid'] = self.feature_frame.apply(lambda x: x['id'] in indices, axis=1)
        self.feature_frame = self.feature_frame[self.feature_frame['valid']]
        self.feature_frame = self.feature_frame.drop(columns=['id', 'valid'])
        self.feature_frame = self.feature_frame.reset_index(drop=True)
        self.feature_mapping = pd.read_csv(feature_mapping_csv, index_col='Feature Type')

        self._preprocess()
        self.parkinsons_frame = self.feature_frame[self.feature_frame['class'] == 1].reset_index(drop=True)
        self.non_parkinsons_frame = self.feature_frame[self.feature_frame['class'] == 0].reset_index(drop=True)
        self.SMOTE = SMOTE
        if self.SMOTE:
            self.numpy_data = self.feature_frame.to_numpy(dtype=np.float32)
            self.numpy_feature = self.numpy_data[:, :-1]
            self.numpy_label = self.numpy_data[:, -1]
            self._oversample()

    def _preprocess(self):
        print("Normalizing data..")
        df = self.feature_frame
        if 'id' in df:
            df.drop(columns=['id'], inplace=True)
        skip_column = ['index', 'gender', 'class']
        columns = list(df.columns)
        columns = [c for c in columns if c not in skip_column]
        for col in columns:
            mean = self.feature_mapping[self.feature_mapping['Features'] == col]['mean'][0]
            std = self.feature_mapping[self.feature_mapping['Features'] == col]['std'][0]
            df[col] = (df[col] - mean)/std
        self.feature_frame = df

    def _oversample(self):
        oversampler = ADASYN(random_state=config.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.numpy_feature, self.numpy_label)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            # random.Random(80476).shuffle(to_keep)
            to_keep.append('class')
            to_keep = ['id'] + to_keep
            self.feature_frame = self.feature_frame[to_keep]

    def __len__(self):
        if self.SMOTE_FLAG:
            # print("Total rows", len(self.numpy_feature))
            return len(self.numpy_feature)
        return len(self.feature_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE_FLAG:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            sample = {'features': features, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample

        datapoint = self.feature_frame.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class GenderDataset(Dataset):

    def __init__(self, dataset_path, indices, feature_mapping_file, feature_score_file=None, max_features=None, SMOTE_FLAG=False, transform=None):
        self.df = pd.read_csv(dataset_path)
        self.df['class'] = self.df.apply(lambda x: 1 if x['label'] == 'male' else 0, axis=1)
        self.df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
        self.feature_score_file = feature_score_file
        self.mapping_file = feature_mapping_file
        self.max_features = max_features
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.normalize()
        self.SMOTE_FLAG = SMOTE_FLAG
        if self.SMOTE_FLAG:
            self.data = self.df.to_numpy(dtype=np.float32)
            self.features, self.labels = self.data[:, :-1], self.data[:, -1]
            self._oversample()
        self.transform = transform
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
        oversampler = ADASYN(random_state=config.SMOTE_SEED)
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
            to_keep = [col for col, _ in scores[:self.max_features]]
            # random.Random(80476).shuffle(to_keep)
            to_keep.append('class')
            # to_keep = ['id'] + to_keep
            self.df = self.df[to_keep]

    def __len__(self):
        if self.SMOTE_FLAG:
            # print("Total rows", len(self.numpy_feature))
            return len(self.numpy_feature)
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE_FLAG:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            sample = {'features': features, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample

        datapoint = self.df.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Pv2Dataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, csv_file, indices, augment=None, max_features=None, feature_score_file=None, feature_mapping_file=None,
                 transform=None, SMOTE_FLAG=False):
        """
        Args:
            csv_file (Path): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(csv_file)
        if 'UPDRS' in self.feature_frame:
            self.feature_frame.drop(columns=['UPDRS'], inplace=True)
        self.feature_score_file = feature_score_file
        self.max_features = max_features if max_features else self.feature_frame.shape[1] - 2 # minus 2 for id and class
        self._filter_feature()
        self.feature_frame['valid'] = self.feature_frame.apply(lambda x: x['Subject id'] in indices, axis=1)
        self.feature_frame = self.feature_frame[self.feature_frame['valid']]
        self.feature_frame = self.feature_frame.drop(columns=['Subject id', 'valid'])
        self.feature_frame = self.feature_frame.reset_index(drop=True)
        self.augment = augment
        self.mapping_file = feature_mapping_file
        self.skip_column = ['class']
        # self.feature_mapping = pd.read_csv(feature_mapping_csv, index_col='Feature Type')
        self.normalize()
        self.parkinsons_frame = self.feature_frame[self.feature_frame['class'] == 1].reset_index(drop=True)
        self.non_parkinsons_frame = self.feature_frame[self.feature_frame['class'] == 0].reset_index(drop=True)
        self.SMOTE_FLAG = SMOTE_FLAG
        if self.SMOTE_FLAG:
            self.numpy_data = self.feature_frame.to_numpy(dtype=np.float32)
            self.numpy_feature = self.numpy_data[:, :-1]
            self.numpy_label = self.numpy_data[:, -1]
            self._oversample()
        self.transform = transform
        # self._group()

    def normalize(self):
        with open(self.mapping_file, 'r', encoding='utf8') as f:
            mapping = json.load(f)
        columns = list(self.feature_frame.columns)
        columns = [c for c in columns if c not in self.skip_column]
        for col in columns:
            self.feature_frame[col] = (self.feature_frame[col] - mapping[col]['mean']) / mapping[col]['std']
        return

    def _oversample(self):
        oversampler = ADASYN(random_state=config.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.numpy_feature, self.numpy_label)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            # random.Random(80476).shuffle(to_keep)
            to_keep.append('class')
            to_keep = ['Subject id'] + to_keep
            self.feature_frame = self.feature_frame[to_keep]

    # def _group(self):
    #     df = self.feature_frame
    #     feature_map_df = self.feature_mapping
    #     for feature in self.select_feature:
    #         req_feat = feature_map_df.loc[feature]['Features'].tolist()
    #         self.grouped_frame.append((feature, df[req_feat].copy()))

    def _augmentor(self, data):
        selected_feature = data.sample(self.augment)
        for f in selected_feature.index:
            if data['class'] == 1:
                alter_value = random.choice(self.parkinsons_frame[f])
            else:
                alter_value = random.choice(self.non_parkinsons_frame[f])
            data[f] = alter_value
        return data

    def __len__(self):
        if self.SMOTE_FLAG:
            # print("Total rows", len(self.numpy_feature))
            return len(self.numpy_feature)
        return len(self.feature_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE_FLAG:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            sample = {'features': features, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample

        datapoint = self.feature_frame.iloc[idx]
        if self.augment:
            datapoint = self._augmentor(datapoint.copy())
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class PilipinoDataset(Dataset):

    def __init__(self, dataset_path, indices, feature_mapping_file, feature_score_file=None, max_features=None, SMOTE_FLAG=False, transform=None):
        self.df = pd.read_csv(dataset_path)
        self.feature_score_file = feature_score_file
        self.mapping_file = feature_mapping_file
        self.max_features = max_features
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.normalize()
        self.transform = transform
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
        oversampler = ADASYN(random_state=config.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.features, self.labels)

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
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ArabicDataset(Dataset):

    def __init__(self, dataset_path, indices, feature_score_file=None, max_features=None, SMOTE_FLAG=False, transform=None):
        self.df = pd.read_csv(dataset_path)
        cmap = ['happy', 'surprised', 'angry']
        self.df['class'] = self.df.apply(lambda x: cmap.index(x['Emotion ']), axis=1)
        self.df.drop(columns=['name', 'Emotion ', 'Type'], inplace=True)
        self.feature_score_file = feature_score_file
        self.max_features = max_features
        self._filter_feature()
        self.df = self.df.iloc[indices]
        self.skip_column = ['class']
        self.SMOTE_FLAG = SMOTE_FLAG
        if self.SMOTE_FLAG:
            self.data = self.df.to_numpy(dtype=np.float32)
            self.features, self.labels = self.data[:, :-1], self.data[:, -1]
            self._oversample()
        self.transform = transform
        return

    # def normalize(self):
    #     with open(self.mapping_file, 'r', encoding='utf8') as f:
    #         mapping = json.load(f)
    #     columns = list(self.df.columns)
    #     columns = [c for c in columns if c not in self.skip_column]
    #     for col in columns:
    #         self.df[col] = (self.df[col] - mapping[col]['mean']) / mapping[col]['std']
    #     return

    def _oversample(self):
        oversampler = ADASYN(random_state=config.SMOTE_SEED)
        self.numpy_feature, self.numpy_label = oversampler.fit_resample(self.features, self.labels)

    def _filter_feature(self):
        if self.feature_score_file:
            with open(self.feature_score_file, 'r', encoding='utf8') as handle:
                scores = list(json.load(handle).items())
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            to_keep = [col for col, _ in scores[:self.max_features]]
            to_keep.append('class')
            self.df = self.df[to_keep]

    def __len__(self):
        if self.SMOTE_FLAG:
            # print("Total rows", len(self.numpy_feature))
            return len(self.numpy_feature)
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.SMOTE_FLAG:
            features = self.numpy_feature[idx]
            label = self.numpy_label[idx]
            sample = {'features': features, 'label': label}
            if self.transform:
                sample = self.transform(sample)
            return sample

        datapoint = self.df.iloc[idx]
        datapoint = datapoint.to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']
        # label = 1.0 if label == 1.0 else 0.0 # invert class
        return {'features': torch.from_numpy(features),
                'label': torch.Tensor([label])}

class ToTensor_categorical(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']
        # label = 1.0 if label == 1.0 else 0.0 # invert class
        # lab = [0.0, 0.0, 0.0]
        # lab[int(label)] = 1.0
        return {'features': torch.from_numpy(features),
                'label': torch.Tensor([label])}


# class ToTensorGroup(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         features, label = sample['features'], sample['label']
#         return {'features': torch.from_numpy(features),
#                 'label': torch.Tensor([label])}


if __name__ == "__main__":

    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    dataset_name = config['Exp Details']['Dataset']
    data_split_file = config['datasets'][dataset_name]['split']
    with open(data_split_file, 'r', encoding='utf8') as f:
        split_detail = json.load(f)

    dataset_map = {'Parkisions': ParkinsonsDataset, 'Gender': GenderDataset}

    dataset = Pv2Dataset(split_detail['train_1'], config)
    train_dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['features'].size(), sample_batched['label'].size())
        break