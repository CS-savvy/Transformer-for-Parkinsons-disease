import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ParkinsonsDataset(Dataset):
    """parkinson's dataset."""

    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with features and label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.feature_frame = pd.read_csv(csv_file, skiprows=[0])
        self.transform = transform

    def __len__(self):
        return len(self.feature_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint = self.feature_frame.iloc[idx].to_numpy(dtype=np.float32)
        features, label = datapoint[:-1], datapoint[-1]
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


if __name__ == "__main__":

    parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv', transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(parkinson_dataset, batch_size=4, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['features'].size(), sample_batched['label'].size())

        break