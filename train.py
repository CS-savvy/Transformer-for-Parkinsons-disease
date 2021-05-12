import torch
from torch.utils.data import DataLoader
from network.Model import Transformer
from DataLoader import ParkinsonsDataset, ToTensor
# from sklearn.metrics import recall_score
import torch.nn as nn
# from focal_loss.focal_loss import FocalLoss
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("using device : ", device)

def train(model, train_dataloader, epochs):
    model = model.to(device)
    # writer = SummaryWriter(comment=f"LR_{config.LR}_BATCH_{config.BATCH_SIZE}")
    criterion = nn.BCELoss()
    # criterion = FocalLoss(alpha=2, gamma=5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train_loss_history = []
    train_accuracy_history = []
    # recall_history = []
    # val_loss_history = []
    # val_accuracy_history = []
    # val_recall_history = []
    # val_max_score = 0.0
    for epoch in range(1, epochs + 1):

        train_loss = 0.0
        train_accuracy = 0.0
        y_preds = []
        y_labels = []

        for batch_data in tqdm(train_dataloader, desc="Epoch %s" % epoch):

            features = batch_data['features']
            labels = batch_data['label']
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.round()
            y_preds.extend(list(preds.cpu().detach().numpy().reshape(1, -1)[0]))
            y_labels.extend(list(labels.cpu().detach().numpy().reshape(1, -1)[0]))

            train_accuracy += torch.sum(preds == labels).item()
            train_loss += loss.item()

        else:
            train_loss = train_loss / train_dataloader.sampler.num_samples
            train_accuracy = train_accuracy / train_dataloader.sampler.num_samples
            # recall = recall_score(y_labels, y_preds)
            train_loss_history.append(train_loss)
            train_accuracy_history.append(train_accuracy)
            # recall_history.append(recall)

            print(f"Metrics for Epoch {epoch}: Train Loss:{round(train_loss, 4)} \
                    Train Accuracy: {round(train_accuracy, 4)}")

    return {
        'training_loss': train_loss_history,
        'training_accuracy': train_accuracy_history,
    }


if __name__ == '__main__':

    # split name must equal to split filename eg: for train.txt -> train
    parkinson_dataset = ParkinsonsDataset(csv_file='data/pd_speech_features.csv', transform=transforms.Compose([ToTensor()]))
    train_data = DataLoader(parkinson_dataset, batch_size=4, shuffle=True)

    epoch = 10
    embedding_dim = 64
    encoder_layer = 6
    attention_head = 1

    model = Transformer(embedding_dim, encoder_layer, attention_head, dropout=0.1, feature_length=754)
    history = train(model, train_data, epoch)
    print(history)