import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import config
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ParkinsonsDataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, csv_file, max_length=432, select_feature=None, feature_mapping_csv=None, shuffle=False, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(csv_file, skiprows=[0])
        self.shuffle = shuffle
        self.feature_mapping = pd.read_csv(feature_mapping_csv, index_col='Feature Type')
        self.select_feature = select_feature
        self.max_length = max_length
        self.grouped_frame = []
        self._preprocess()
        # self._filter_feature()
        self.transform = transform
        self._group()

    def _preprocess(self):
        print("Normalizing data..")
        df = self.feature_frame
        df.drop(columns=['id'], inplace=True)
        skip_column = ['gender', 'class']
        columns = list(df.columns)
        columns = [c for c in columns if c not in skip_column]
        for col in columns:
            df[col] = (df[col] - df[col].mean())/df[col].std(ddof=0)

        if self.shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        self.feature_frame = df

    def _filter_feature(self):
        if self.select_feature:
            feature_map_df = self.feature_mapping
            req_feat = feature_map_df.loc[self.select_feature]['Features'].tolist()
            self.feature_frame = self.feature_frame[req_feat + ['class']]

    def _group(self):
        df = self.feature_frame
        feature_map_df = self.feature_mapping
        for feature in self.select_feature:
            req_feat = feature_map_df.loc[feature]['Features'].tolist()
            self.grouped_frame.append((feature, df[req_feat].copy()))

    def __len__(self):
        return len(self.feature_frame)

    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()
    #
    #     datapoint = self.feature_frame.iloc[idx].to_numpy(dtype=np.float32)
    #     features, label = datapoint[:-1], datapoint[-1]
    #     sample = {'features': features, 'label': label}
    #     if self.transform:
    #         sample = self.transform(sample)
    #
    #     return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        datapoint = self.feature_frame.iloc[idx].to_numpy(dtype=np.float32)
        label = datapoint[-1]
        features = []
        for feature_type, df in self.grouped_frame:
            datapoint = df.iloc[idx].to_numpy(dtype=np.float32)
            datapoint = np.pad(datapoint, (0, self.max_length - datapoint.shape[0]), 'constant')
            features.append(datapoint)
        features = np.array(features)
        sample = {'features': features, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']
        return {'features': torch.from_numpy(features),
                'label': torch.Tensor([label])}


class ToTensorGroup(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        features, label = sample['features'], sample['label']
        return {'features': torch.from_numpy(features),
                'label': torch.Tensor([label])}


if __name__ == "__main__":

    parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv',
                                          select_feature=config.FEATURES,
                                          feature_mapping_csv='data/feature_mapping.csv',
                                          transform=transforms.Compose([ToTensorGroup()]))

    indexes = list(range(parkinson_dataset.__len__()))
    train_indices, val_indices = indexes[:660], indexes[660:]
    train_set = torch.utils.data.dataset.Subset(parkinson_dataset, train_indices)
    val_set = torch.utils.data.dataset.Subset(parkinson_dataset, val_indices)

    train_dataloader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)
    for i_batch, sample_batched in enumerate(train_dataloader):
        print(i_batch, sample_batched['features'].size(), sample_batched['label'].size())
        break