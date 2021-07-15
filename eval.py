import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from network.Model import Transformer, MLP, FeatureEmbedMLP, TransformerGroup, DeepMLP, ConvModel
from DataLoader import ParkinsonsDataset, ToTensor
from sklearn import metrics
import torch.nn as nn
from focal_loss.focal_loss import FocalLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import torch.nn.functional as F
import config
from pathlib import Path
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device : ", device)

dataset_path = Path("data/pd_speech_features.csv")
feature_mapping_file = Path("data/feature_details.csv")


def evaluate(model_dir, threshold=0.5, split_details=None):
    best_model = None
    max_auc_score = 0
    results = []
    val_aucs = []
    for i in range(1, 11):
        model = Transformer(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
                            dropout=config.DROPOUT, feature_length=config.MAX_FEATURE)
        # model = ConvModel(16, 32, 1024, 512)
        checkpoint = torch.load(model_dir / f"model_k_fold_{i}.pt")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        val_indices = split_details[f'val_{i}']
        val_set = ParkinsonsDataset(dataset_path, val_indices, feature_score_file='data/imp_feature_xgboost.pkl',
                                    max_features=config.MAX_FEATURE, feature_mapping_csv=feature_mapping_file,
                                    transform=transforms.Compose([ToTensor()]))
        dataloader = DataLoader(val_set, batch_size=len(val_indices), shuffle=False)
        y_preds = []
        y_scores = []
        y_labels = []
        all_uids = []
        tp = 0
        with torch.no_grad():
            for batch_data in dataloader:
                uids = batch_data['uid']
                features = batch_data['features']
                labels = batch_data['label']
                features = features.to(device)
                print(features.shape)
                labels = labels.to(device)

                outputs = model(features)
                pred_score = F.sigmoid(outputs)
                # preds = pred_score.round()
                preds = (pred_score > threshold) * 1.0
                y_scores.extend(list(pred_score.cpu().detach().numpy().reshape(1, -1)[0]))
                y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
                y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))
                all_uids.extend(list(uids.numpy().reshape(1, -1)[0]))
                tp += torch.sum(preds == labels).item()

            accuracy = tp / len(val_indices)
            precision = metrics.precision_score(y_labels, y_preds)
            recall = metrics.recall_score(y_labels, y_preds)
            roc_auc = metrics.roc_auc_score(y_labels, y_scores)
            f1 = metrics.f1_score(y_labels, y_preds)
            print(f"{i} - Accuracy : {accuracy} | precision : {precision} | Recall : {recall} | F1 : {f1} | ROC : {roc_auc}")
            val_aucs.append(roc_auc)
            if roc_auc > max_auc_score:
                max_auc_score = roc_auc
                best_model = i

        results.extend([(u, l, p, s, l == p) for u, l, p, s in zip(all_uids, y_labels, y_preds, y_scores)])
    print(f"Avg AUC accuracy - ", sum(val_aucs)/ len(val_aucs))
    print(f"Best Model: {best_model} having AUC {max_auc_score}")
    test_indices = split_details['test']
    test_set = ParkinsonsDataset(dataset_path, test_indices, feature_score_file='data/imp_feature_xgboost.pkl',
                                 max_features=config.MAX_FEATURE, feature_mapping_csv=feature_mapping_file,
                                 transform=transforms.Compose([ToTensor()]))
    test_dataloader = DataLoader(test_set, batch_size=len(test_indices), shuffle=False)
    model = Transformer(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
                        dropout=config.DROPOUT, feature_length=config.MAX_FEATURE)
    # model = ConvModel(16, 32, 1024, 512)
    checkpoint = torch.load(model_dir / f"model_k_fold_{best_model}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    y_preds = []
    y_scores = []
    y_labels = []
    all_uids = []
    tp = 0
    with torch.no_grad():
        for batch_data in test_dataloader:
            uids = batch_data['uid']
            features = batch_data['features']
            labels = batch_data['label']
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            pred_score = F.sigmoid(outputs)
            # preds = pred_score.round()
            preds = (pred_score > threshold) * 1.0
            y_scores.extend(list(pred_score.cpu().detach().numpy().reshape(1, -1)[0]))
            y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))
            all_uids.extend(list(uids.numpy().reshape(1, -1)[0]))
            tp += torch.sum(preds == labels).item()

        accuracy = tp / len(test_indices)
        precision = metrics.precision_score(y_labels, y_preds)
        recall = metrics.recall_score(y_labels, y_preds)
        f1 = metrics.f1_score(y_labels, y_preds)
        roc_auc = metrics.roc_auc_score(y_labels, y_scores)
        print(f"Test - Accuracy : {accuracy} | precision : {precision} | Recall : {recall} | F1 : {f1} | ROC-AUC : {roc_auc}")
        results.extend([(u, l, p, s, l == p) for u, l, p, s in zip(all_uids, y_labels, y_preds, y_scores)])

    results = sorted(results, key=lambda x: x[0])
    result_df = pd.DataFrame(results, columns=["UID", 'True Label', 'Prediction', 'Score', 'Match'])
    result_df.to_csv(config.MODEL_DIR / f"eval_result.csv", index=False)
    return accuracy


if __name__ == "__main__":

    with open("data/split_details.json", 'r', encoding='utf8') as f:
        split_details = json.load(f)

    # parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv',
    #                                       select_feature=config.FEATURES,
    #                                       feature_mapping_csv='data/feature_mapping.csv',
    #                                       transform=transforms.Compose([ToTensor()]))

    threshes = np.arange(0, 1, 0.05)
    # all_accuracy = {}
    # for i in threshes:
    #     print(i)
    #     result = evaluate(config.MODEL_DIR, threshold=i, split_details=split_details)
    #     all_accuracy[i] = result
    # print(all_accuracy)

    result = evaluate(config.MODEL_DIR, threshold=0.5, split_details=split_details)