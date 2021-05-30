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


def train(model, train_data, val_data, id=None):

    model = model.to(device)
    writer = SummaryWriter(comment=f"LR_{config.LR}_BATCH_{config.BATCH_SIZE}")
    criterion = nn.BCEWithLogitsLoss()
    # criterion = FocalLoss(alpha=2, gamma=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)

    train_loss_history = []
    train_accuracy_history = []
    train_precision_history = []
    train_recall_history = []

    val_loss_history = []
    val_accuracy_history = []
    val_precision_history = []
    val_recall_history = []

    max_val_accuracy = 0

    sample_data = train_data.__iter__().next()
    features = sample_data['features']
    features = features.to(device)
    writer.add_graph(model, input_to_model=features)
    print("Training started ...")
    for epoch in range(1, config.EPOCH + 1):

        train_loss_mini_batch = []
        train_accuracy_mini_batch = []
        y_preds = []
        y_labels = []
        model.train()
        for batch_data in train_data:
            # get the inputs; data is a list of [inputs, labels]
            features = batch_data['features']
            features = features.to(device)
            labels = batch_data['label']
            # features = features.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = F.sigmoid(outputs).round()
            y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))

            train_accuracy_mini_batch.append(torch.sum(preds == labels).item())
            train_loss_mini_batch.append(loss.item())
            # print("Mini batch loss", loss.item())
        else:
            train_loss = sum(train_loss_mini_batch) / len(train_loss_mini_batch)
            train_accuracy = sum(train_accuracy_mini_batch) / train_data.sampler.num_samples
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            train_precision = metrics.precision_score(y_labels, y_preds)
            train_recall = metrics.recall_score(y_labels, y_preds)
            train_precision_history.append(train_precision)
            train_recall_history.append(train_recall)

            val_loss_mini_batch = []
            val_accuracy_mini_batch = []
            val_y_preds = []
            val_y_labels = []
            val_uids = []
            model.eval()
            with torch.no_grad():
                for batch_data in val_data:
                    uids = batch_data['uid']
                    features = batch_data['features']
                    labels = batch_data['label']
                    features = features.to(device)
                    labels = labels.to(device)

                    outputs = model(features)
                    loss = criterion(outputs, labels)
                    preds = F.sigmoid(outputs).round()

                    val_y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
                    val_y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))
                    val_uids.extend(list(uids.numpy().reshape(1, -1)[0]))
                    val_accuracy_mini_batch.append(torch.sum(preds == labels).item())
                    val_loss_mini_batch.append(loss.item())

            val_loss = sum(val_loss_mini_batch) / len(val_loss_mini_batch)
            val_accuracy = sum(val_accuracy_mini_batch) / val_data.sampler.num_samples
            val_loss_history.append(val_loss)
            val_accuracy_history.append(val_accuracy)
            val_precision = metrics.precision_score(val_y_labels, val_y_preds)
            val_recall = metrics.recall_score(val_y_labels, val_y_preds)
            val_precision_history.append(val_precision)
            val_recall_history.append(val_recall)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Precision/train', train_precision, epoch)
            writer.add_scalar('Recall/train', train_recall, epoch)

            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Accuracy/validation', val_accuracy, epoch)
            writer.add_scalar('Precision/validation', val_precision, epoch)
            writer.add_scalar('Recall/validation', val_precision, epoch)

            print(f"Metrics for Epoch {epoch}: Train Loss:{round(train_loss, 8)} \
                    Train Accuracy: {round(train_accuracy, 8)} Train Precision: {round(train_precision, 8)} \
                    Train Recall: {round(train_recall, 8)}")
            print(f"Metrics for Epoch {epoch}: val Loss:{round(val_loss, 8)} \
                    Val Accuracy: {round(val_accuracy, 8)} Val Precision: {round(val_precision, 8)} \
                    Val Recall: {round(val_recall, 8)}")

            if val_accuracy > max_val_accuracy:
                print("Saving model ....")
                max_val_accuracy = val_accuracy
                model_content = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                                 'optimizer_state_dict': optimizer.state_dict(),
                                 'val_acc': val_accuracy, 'train_acc': train_accuracy}
                torch.save(model_content, config.MODEL_DIR / f"model_k_fold_{id}.pt")

    writer.flush()
    writer.close()
    return {
        'training_loss': train_loss_history,
        'training_accuracy': train_accuracy_history,
        'train_recall': train_recall_history,
        'train_precision': train_precision_history,
        'val_loss': val_loss_history,
        'val_accuracy': val_accuracy_history,
        'val_recall': val_recall_history,
        'val_precision': val_precision_history,
    }


if __name__ == '__main__':

    parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv',
                                          select_feature=config.FEATURES,
                                          feature_mapping_csv='data/feature_mapping.csv',
                                          transform=transforms.Compose([ToTensor()]))

    k_fold = 10
    model_histories = []

    with open("data/split_details.json", 'r', encoding='utf8') as f:
        split_detail = json.load(f)

    for i in range(1, k_fold+1):
        print(f"Started {i} of {k_fold}-fold training .... ")
        train_set = torch.utils.data.dataset.Subset(parkinson_dataset, split_detail[f'train_{i}'])
        val_set = torch.utils.data.dataset.Subset(parkinson_dataset, split_detail[f'val_{i}'])

        train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
        val_dataloader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=True)

        # tf_model = TransformerGroup(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
        #                        dropout=config.DROPOUT, feature_set=[21, 3, 4, 4, 22, 84, 182, 432])

        # tf_model = Transformer(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
        #                        dropout=config.DROPOUT, feature_length=753)

        tf_model = MLP(config.EMBEDDING_DIM, config.ENCODER_STACK, config.ATTENTION_HEAD,
                                    dropout=config.DROPOUT, feature_length=753)

        history = train(tf_model, train_dataloader, val_dataloader, id=i)
        print(history)
        model_histories.append(history)

    max_val_accuracies = [max(h['val_accuracy']) for h in model_histories]
    print(f"Average val accuracy across {k_fold}-Fold: {np.average(max_val_accuracies)}")
