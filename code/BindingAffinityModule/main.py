from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import torch
import math
import time
import json
import yaml

# my code
from utils import set_up_exp_folder, str2bool, update_metrics
from dataset import BindingDataset, collate_fn
from trainer import train, evaluate, return_values
from model import BindingModel2


def main(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    checkpoint_dir = set_up_exp_folder(args.result_path)
    args.results_folder = checkpoint_dir
    with open(checkpoint_dir + 'config.json', 'w') as f:
        json.dump(vars(args), f)

    with open(config['path2uniprot2map'], 'rb') as f:
        uniprot2intmap = pickle.load(f)
    reverse_identifier_dict = {v: k for k, v in uniprot2intmap.items()}

    train_data = pd.read_csv(f'../datasets/{args.data_split}/train_rs_{args.seed}_{args.data_split}.csv')
    valid_data = pd.read_csv(f'../datasets/{args.data_split}/valid_rs_{args.seed}_{args.data_split}.csv')
    test_data = pd.read_csv(f'../datasets/{args.data_split}/test_rs_{args.seed}_{args.data_split}.csv')

    test_data_copy = test_data

    max_length = config['max_length']

    train_loader = DataLoader(BindingDataset(train_data, max_length, args.new_emb_flag), batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, max_length, uniprot2intmap))
    valid_loader = DataLoader(BindingDataset(valid_data, max_length, args.new_emb_flag), batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, max_length, uniprot2intmap))
    test_loader = DataLoader(BindingDataset(test_data, max_length, args.new_emb_flag), batch_size=args.batch_size, shuffle=False, collate_fn=lambda batch: collate_fn(batch, max_length, uniprot2intmap))

    model = BindingModel2(num_layer=config['num_layers'],
                         emb_dim=config['chem_embed_dim'],
                         dropout=args.dropout,
                         gnn_type=config['gnn_type'],
                         combined_dim=config['combined_dim']).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, amsgrad = True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)


    best_mse = np.inf
    metrics_dict =    { 'train_loss': [], 'train_r2': [], 'train_mae': [], 'train_mse': [], 'train_rmse': [],
    'train_pearsonr': [], 'train_spearmanr': [],

    # Validation metrics
    'valid_loss': [], 'valid_r2': [], 'valid_mae': [], 'valid_mse': [], 'valid_rmse': [],
    'valid_pearsonr': [], 'valid_spearmanr': [], 'learning_rate': [],

    # Test metrics
    'test_loss': [], 'test_r2': [], 'test_mae': [], 'test_mse': [], 'test_rmse': [],
    'test_pearsonr': [], 'test_spearmanr': []
    }

    print(model)

    print('------------------------------------------------------Starting Training------------------------------------------------------')
    for epoch in range(1, args.num_epochs + 1):
        print(f"Running Epoch {epoch}/{args.num_epochs}")
        stime = time.time()
        train_metrics_1 = train(model, train_loader, optimizer, criterion, device)
        # evaluating train, test and valid sets
        train_metrics = evaluate(model, train_loader, criterion, device)
        update_metrics(metrics_dict, 'train', train_metrics)
        valid_metrics = evaluate(model, valid_loader, criterion, device)
        update_metrics(metrics_dict, 'valid', valid_metrics)
        test_metrics = evaluate(model, test_loader, criterion, device)
        update_metrics(metrics_dict, 'test', test_metrics)
        metrics_dict['learning_rate'].append(optimizer.param_groups[0]['lr'])
        df_metrics = pd.DataFrame(metrics_dict)
        df_metrics.to_csv(checkpoint_dir + 'metrics.csv')
        scheduler.step(epoch)
        if best_mse > valid_metrics['mse']:
            print('Valid MSE is greater!')
            best_mse = valid_metrics['mse']
            torch.save(model.state_dict(), checkpoint_dir+"best_model.pth")
            y_true, y_pred = return_values(model, test_loader, criterion, device)
            test_data_copy['Actual Values'] = y_true
            test_data_copy['Predictions'] = y_pred
            test_data_copy.to_csv(checkpoint_dir + 'predictions_of_best_model.csv')

        print(f"Training Set \n - R^2: {train_metrics['r2']:.4f}\n MAE: {train_metrics['mae']:.4f}\n MSE: {train_metrics['mse']:.4f}\n RMSE: {train_metrics['rmse']:.4f}\n Pearson Correlation Coefficient: {train_metrics['pearsonr']:.4f}\n Spearman's Rank Correlation Coefficient: {train_metrics['spearmanr']:.4f}")
        print(f"Validation Set - R^2: {valid_metrics['r2']:.4f}\n MAE: {valid_metrics['mae']:.4f}\n MSE: {valid_metrics['mse']:.4f}\n RMSE: {valid_metrics['rmse']:.4f}\n Pearson Correlation Coefficient: {valid_metrics['pearsonr']:.4f}\n Spearman's Rank Correlation Coefficient: {valid_metrics['spearmanr']:.4f}")
        etime = time.time()
        print(f"Epoch Training Time: {etime - stime:.2f} seconds")

if __name__ == '__main__':
    print('Binding Affinity Prediction Starting........')

    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', type=str, default='../results/logs/', help='path to save best model')
    # hyperparameters
    parser.add_argument('--data_split', type=str, default='random', help='type of data split [random, scaffold, pfam]')
    parser.add_argument('--new_emb_flag', type=str2bool, nargs='?', const=False, default=True, help='For using ESM2 or ESMFold bookings')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epochs for training')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability for model')


    args = parser.parse_args()

    main(args)
