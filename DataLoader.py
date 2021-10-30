import torch
import config
import pandas as pd
import numpy as np
import random
random.seed(config.PYTHON_SEED)
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
import config
# Ignore warnings
import warnings
import pickle
import json
warnings.filterwarnings("ignore")


class ParkinsonsDataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, csv_file, indices, augment=None, max_features=None, feature_score_file=None, feature_mapping_csv=None,
                 transform=None, SMOTE_FLAG=False):
        """
        Args:
            csv_file (Path): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(csv_file, skiprows=[0])
        self.feature_score_file = feature_score_file
        self.max_features = max_features if max_features else self.feature_frame.shape[1] - 2 # minus 2 for id and class
        self._filter_feature()
        self.feature_frame['valid'] = self.feature_frame.apply(lambda x: x['id'] in indices, axis=1)
        self.feature_frame = self.feature_frame[self.feature_frame['valid']]
        self.feature_frame = self.feature_frame.drop(columns=['id', 'valid'])
        self.feature_frame = self.feature_frame.reset_index(drop=True)
        self.augment = augment
        self.feature_mapping = pd.read_csv(feature_mapping_csv, index_col='Feature Type')

        # self.grouped_frame = []
        self._preprocess()
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

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #     datapoint = self.feature_frame.iloc[idx].to_numpy(dtype=np.float32)
    #     label = datapoint[-1]
    #     features = []
    #     for feature_type, df in self.grouped_frame:
    #         datapoint = df.iloc[idx].to_numpy(dtype=np.float32)
    #         datapoint = np.pad(datapoint, (0, self.max_length - datapoint.shape[0]), 'constant')
    #         features.append(datapoint)
    #     features = np.array(features)
    #     sample = {'features': features, 'label': label}
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']
        # label = 1.0 if label == 1.0 else 0.0 # invert class
        return {'features': torch.from_numpy(features),
                'label': torch.Tensor([label])}


# class ToTensorGroup(object):
#     """Convert ndarrays in sample to Tensors."""
#
#     def __call__(self, sample):
#         features, label = sample['features'], sample['label']
#         return {'features': torch.from_numpy(features),
#                 'label': torch.Tensor([label])}


# if __name__ == "__main__":
#
#     parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv',
#                                           select_feature=config.FEATURES,
#                                           feature_mapping_csv='data/feature_mapping.csv',
#                                           transform=transforms.Compose([ToTensor)))
#
#     indexes = list(range(parkinson_dataset.__len__()))
#     train_indices, val_indices = indexes[:660], indexes[660:]
#     train_set = torch.utils.data.dataset.Subset(parkinson_dataset, train_indices)
#     val_set = torch.utils.data.dataset.Subset(parkinson_dataset, val_indices)
#
#     train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
#     val_dataloader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)
#     for i_batch, sample_batched in enumerate(train_dataloader):
#         print(i_batch, sample_batched['features'].size(), sample_batched['label'].size())
#         break