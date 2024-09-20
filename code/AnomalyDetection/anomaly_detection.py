from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import sklearn
import torch
import json

from utils import set_up_exp_folder, load_and_split_data, str2bool, SimpleMLP, evaluate_model, create_df_results

def main(args):

    checkpoint_dir = set_up_exp_folder(args.result_path)
    args.results_folder = checkpoint_dir

    with open(checkpoint_dir + 'config.json', 'w') as f:
        json.dump(vars(args), f)

    train_df = pd.read_csv(f'../embeddings/datasets/md_dataset_train_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')
    test_df = pd.read_csv(f'../embeddings/datasets/md_dataset_test_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')
    valid_df = pd.read_csv(f'../embeddings/datasets/md_dataset_valid_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')

    X_train, y_train = load_and_split_data(train_df, args)
    X_test, y_test = load_and_split_data(test_df, args)
    X_valid, y_valid = load_and_split_data(valid_df, args)

    print('Starting Training for Anomaly Detection.....')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_valid = scaler.transform(X_valid)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.to_numpy()[:, None], dtype=torch.float32)
    test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.to_numpy()[:, None], dtype=torch.float32)
    valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid.to_numpy()[:, None], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_tensor, y_train_tensor), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_tensor, y_valid_tensor), batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleMLP(input_size=X_train.shape[1]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    loss_fn = nn.MSELoss()
    best_validation_loss = np.inf
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
            scheduler.step(epoch + epoch / len(train_loader))
        avg_train_loss = total_loss / len(train_loader.dataset)
        val_loss = evaluate_model(model, valid_loader, loss_fn, device)
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model = model
        print(f'Epoch {epoch+1}/{args.epochs}: Loss {total_loss/len(train_loader.dataset)}')

    model = best_model
    model.eval()
    with torch.no_grad():
        train_pred = model(train_tensor.to(device))
        train_pred = train_pred.cpu().numpy().squeeze()
        test_pred = model(test_tensor.to(device))
        test_pred = test_pred.cpu().numpy().squeeze()
        valid_pred = model(valid_tensor.to(device))
        valid_pred = valid_pred.cpu().numpy().squeeze()

    create_df_results(train_pred, train_df, 'train', checkpoint_dir)
    create_df_results(valid_pred, valid_df, 'valid', checkpoint_dir)
    create_df_results(test_pred, test_df, 'test', checkpoint_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training model')
    parser.add_argument('--result_path', type=str, default='../results/logs_md/', help='path to save best model')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters for KMeans')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training for MLP')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations for KMeans')
    parser.add_argument('--scaling', type=str2bool, nargs='?', const=False, default=True, help='Scaling Embeddings for better clustering')
    parser.add_argument('--data_split', type=str, default='random', help='select from [random, scaffold]')
    parser.add_argument('--seed', type=int, default=42, help='Select Seed, keep the seed same as training to ensure proper embedding extraction')

    args = parser.parse_args()
    main(args)
