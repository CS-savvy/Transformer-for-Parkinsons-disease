import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from network.Model import Transformer, MLP, FeatureEmbedMLP, TransformerGroup
from DataLoader import ParkinsonsDataset, ToTensor
from sklearn import metrics
import torch.nn as nn
from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import config

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device : ", device)


def evaluate(model_dir, data, split_details=None):

    best_model = None
    max_val_accuracy = 0
    for i in range(1, 11):
        model = MLP(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
                    dropout=config.DROPOUT, feature_length=753)
        checkpoint = torch.load(model_dir / f"model_k_fold_{i}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        val_indices = split_details[f'val_{i}']
        val_set = torch.utils.data.dataset.Subset(data, val_indices)
        dataloader = DataLoader(val_set, batch_size=len(val_indices), shuffle=False)
        y_preds = []
        y_labels = []
        all_uids = []
        with torch.no_grad():
            for batch_data in dataloader:
                uids = batch_data['uid']
                features = batch_data['features']
                labels = batch_data['label']
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                preds = F.sigmoid(outputs).round()

                y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
                y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))
                all_uids.extend(list(uids.numpy().reshape(1, -1)[0]))

                accuracy = torch.sum(preds == labels).item() / len(val_indices)
                precision = metrics.precision_score(y_labels, y_preds)
                recall = metrics.recall_score(y_labels, y_preds)
                print(f" Accuracy : {accuracy} | precision : {precision} | Recall : {recall}")

                if accuracy > max_val_accuracy:
                    max_val_accuracy = accuracy
                    best_model = i

    print(f"Best Model: {best_model} having accuracy {max_val_accuracy}")
    test_indices = split_details['test']
    test_set = torch.utils.data.dataset.Subset(data, test_indices)
    test_dataloader = DataLoader(test_set, batch_size=len(test_indices), shuffle=False)
    model = MLP(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
                dropout=config.DROPOUT, feature_length=753)
    checkpoint = torch.load(model_dir / f"model_k_fold_{best_model}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_data in test_dataloader:
            uids = batch_data['uid']
            features = batch_data['features']
            labels = batch_data['label']
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            preds = F.sigmoid(outputs).round()

            y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))
            all_uids.extend(list(uids.numpy().reshape(1, -1)[0]))

            accuracy = torch.sum(preds == labels).item() / len(test_indices)
            precision = metrics.precision_score(y_labels, y_preds)
            recall = metrics.recall_score(y_labels, y_preds)
            print(f"Test - Accuracy : {accuracy} | precision : {precision} | Recall : {recall}")


if __name__ == "__main__":

    with open("data/split_details.json", 'r', encoding='utf8') as f:
        split_details = json.load(f)

    parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv',
                                          select_feature=config.FEATURES,
                                          feature_mapping_csv='data/feature_mapping.csv',
                                          transform=transforms.Compose([ToTensor()]))

    evaluate(config.MODEL_DIR, parkinson_dataset, split_details=split_details)