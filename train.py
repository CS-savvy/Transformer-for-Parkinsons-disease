import json
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from network.Model import ModelManager
from losses import Losses
from datasets import DatasetManager
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import yaml


def eval_network(model, data, criterion, device):
    model.eval()
    loss = 0
    acc = 0
    precision = 0
    recall = 0
    auc = 0

    count = 0
    with torch.no_grad():
        for i_batch, (batched_features, batched_target) in enumerate(data):
            batched_features = batched_features.to(device)
            batched_target = batched_target.to(device)
            logits = model(batched_features)
            network_loss = criterion(logits, batched_target.reshape(-1, 1))
            loss += network_loss.item()
            pred_labels = torch.sigmoid(logits).round().reshape(-1).cpu().detach().numpy()
            targets = batched_target.cpu().detach().numpy()
            acc += np.sum(pred_labels == targets) / len(pred_labels)
            precision += metrics.precision_score(targets, pred_labels)
            recall += metrics.recall_score(targets, pred_labels)
            auc += metrics.roc_auc_score(targets, pred_labels)
            count += 1

    loss /= count
    acc /= count
    precision /= count
    recall /= count
    auc /= count

    return loss, acc, precision, recall, auc


def train_epoch(model, data, optimizer, criterion, device):
    model.train()
    loss = 0
    acc = 0
    precision = 0
    recall = 0
    auc = 0
    count = 0
    for i_batch, (batched_features, batched_target) in enumerate(data):
        batched_features = batched_features.to(device)
        batched_target = batched_target.to(device)
        optimizer.zero_grad()
        logits = model(batched_features)
        network_loss = criterion(logits, batched_target.reshape(-1, 1))
        network_loss.backward()
        optimizer.step()
        loss += network_loss.item()
        pred_labels = torch.sigmoid(logits).round().reshape(-1).cpu().detach().numpy()
        targets = batched_target.cpu().detach().numpy()
        acc += np.sum(pred_labels == targets) / len(pred_labels)
        precision += metrics.precision_score(targets, pred_labels)
        recall += metrics.recall_score(targets, pred_labels)
        auc += metrics.roc_auc_score(targets, pred_labels)
        count += 1

    loss /= count
    acc /= count
    precision /= count
    recall /= count
    auc /= count

    return loss, optimizer, acc, precision, recall, auc


def train_model(cv, params):
    device = params['Device']
    model = params['model'].to(device)
    criterion = Losses().get_criterion(params['Loss'])()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['LR'])
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    writer = params['writer']
    train_data, val_data = params['train_data'], params['val_data']
    sample_features, _ = train_data.__iter__().next()
    sample_features = sample_features.to(device)
    writer.add_graph(model, input_to_model=sample_features)

    max_val_acc = 0

    print("Training started ...")
    for epoch in range(1, params['Epoch'] + 1):

        train_logs = train_epoch(model, train_data, optimizer, criterion, device)
        train_loss, optimizer, train_acc, train_prec, train_rec, train_auc = train_logs

        val_logs = eval_network(model, val_data, criterion, device)
        val_loss, val_acc, val_prec, val_rec, val_auc = val_logs

        writer.add_scalars('train/_loss', {f'cv_{cv}':  train_loss}, epoch)
        writer.add_scalars('train/_accuracy', {f'cv_{cv}': train_acc}, epoch)
        writer.add_scalars('train/_precision', {f'cv_{cv}': train_prec}, epoch)
        writer.add_scalars('train/_recall', {f'cv_{cv}': train_rec}, epoch)
        writer.add_scalars('train/_auc', {f'cv_{cv}': train_auc}, epoch)

        writer.add_scalars('val/_loss', {f'cv_{cv}': val_loss}, epoch)
        writer.add_scalars('val/_accuracy', {f'cv_{cv}': val_acc}, epoch)
        writer.add_scalars('val/_precision', {f'cv_{cv}': val_prec}, epoch)
        writer.add_scalars('val/_recall', {f'cv_{cv}': val_rec}, epoch)
        writer.add_scalars('val/_auc', {f'cv_{cv}': val_auc}, epoch)

        print(f"Metrics for Epoch {epoch}: Train Loss:{round(train_loss, 8)} \
                Train Accuracy: {round(train_acc, 8)} Train Precision: {round(train_prec, 8)} \
                Train Recall: {round(train_rec, 8)}")
        print(f"Metrics for Epoch {epoch}: val Loss:{round(val_loss, 8)} \
                Val Accuracy: {round(val_acc, 8)} Val Precision: {round(val_prec, 8)} \
                Val Recall: {round(val_rec, 8)} Val AUC: {val_auc}")

        if epoch > 3 and val_acc > max_val_acc:
            print("Saving model ....")
            max_val_acc = val_acc
            model_content = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                             'val_auc': val_auc, 'val_acc': val_acc,
                             'train_acc': train_acc}
            torch.save(model_content, params['ModelDir'] / f"model_k_fold_{cv}.pt")


def init(params: dict) -> dict:
    if params['Device'] == 'gpu':
        if torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            print("GPU not available, using cpu only.")
            dev = torch.device('cpu')
    else:
        dev = torch.device('cpu')
    params['Device'] = dev

    random.seed(params['Seed'])
    np.random.seed(params['Seed'])
    torch.manual_seed(params['Seed'])
    if dev.type == 'cuda':
        torch.cuda.manual_seed(params['Seed'])

    model_dir = Path(params['ModelDir'])
    if model_dir.exists():
        print("Model dir already exists - Do you want to overwrite ? - y/n ")
        # if str(input()).lower() == 'n':
        #     exit()
    else:
        model_dir.mkdir(parents=True)
    params['ModelDir'] = model_dir
    return params


def filterConfigByDtype(params, accepted_dtype):
    temp = {}
    for key, val in params.items():
        if type(val) in accepted_dtype:
            temp[key] = val
    return temp


def save_config(params, d_name, m_name):
    output_dir = params['Setup']['ModelDir']
    params['Setup']['ModelDir'] = output_dir.as_posix()
    acceptable_dtype = [int, str, float]
    config_group = {'Dataset': params['Datasets'][d_name], 'Network': params['Networks'][m_name],
                    'Setup': params['Setup'], 'Train': params['Train']}
    final_config = {}
    for key, cnf in config_group.items():
        final_config[key] = filterConfigByDtype(cnf, acceptable_dtype)

    with open(output_dir / 'exp_config.yaml', 'w', encoding='utf8') as f:
        yaml.safe_dump(final_config, f)


if __name__ == '__main__':

    config_file = Path('config.yaml')
    with open(config_file, 'r', encoding='utf8') as f:
        config = yaml.safe_load(f)
    config['Setup'] = init(config['Setup'])
    dataset_name = config['ExpDetails']['Dataset']
    model_name = config['ExpDetails']['Network']
    dataset_config = config['Datasets'][dataset_name]
    print("Dataset details: ", dataset_config)
    data_split_file = dataset_config['Split']
    with open(data_split_file, 'r', encoding='utf8') as f:
        split_detail = json.load(f)

    model_config = config['Networks'][model_name]
    print("Model config", model_config)
    mm = ModelManager()
    # config['Train']['model'] = mm.get_model(model_name)(model_config)
    batch_size = config['Train']['BatchSize']
    dm = DatasetManager()
    dataset = dm.get_dataset(dataset_name)

    config['Train']['writer'] = SummaryWriter(config['Setup']['ModelDir'] / 'events')
    k_fold = dataset_config['K-Fold']

    save_conf = True
    for i in range(1, k_fold+1):
        print(f"Started {i} of {k_fold}-fold training .... ")
        train_set = dataset(split_detail[f'train_{i}'], dataset_config)
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=train_set.collate)
        config["Train"]['train_data'] = train_dataloader
        val_set = dataset(split_detail[f'val_{i}'], dataset_config)
        val_dataloader = DataLoader(val_set, batch_size=val_set.__len__(), shuffle=False, collate_fn=val_set.collate)
        config['Train']['val_data'] = val_dataloader
        model_config['num_features'] = val_set.get_num_feature_length()
        config['Train']['model'] = mm.get_model(model_name)(model_config)
        train_config = config['Train'].copy().update(config['Setup'])
        if save_conf:
            save_config(config, dataset_name, model_name)
            save_conf = False
        train_model(i, train_config)

    config['Train']['writer'].flush()
    config['Train']['writer'].close()
