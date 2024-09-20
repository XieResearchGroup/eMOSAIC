import math
import torch
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    y_true, y_pred= [], []


    for batch in train_loader:
        batch_protein_seqs, batch_lengths, batch_masks, batch_chem_graphs, targets, uniprot_ids_tensor, indicies = [item.to(device) for item in batch]

        optimizer.zero_grad()

        outputs_model = model(batch_protein_seqs, batch_chem_graphs, batch_masks)
        loss = criterion(outputs_model, targets)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        y_true.extend(targets.cpu().detach().numpy())
        y_pred.extend(outputs_model.cpu().detach().numpy())

    avg_loss = total_loss / len(train_loader)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    metrics = {
        'loss': avg_loss,
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearsonr': pearson_corr,
        'spearmanr': spearman_corr}

    return metrics

def evaluate(model, eval_loader, criterion, device):
    model.eval()
    total_loss = 0
    y_true, y_pred= [], []

    with torch.no_grad():
        for batch in eval_loader:
            batch_protein_seqs, batch_lengths, batch_masks, batch_chem_graphs, targets, uniprot_ids_tensor, indicies = [item.to(device) for item in batch]

            outputs_model = model(batch_protein_seqs, batch_chem_graphs, batch_masks)
            loss = criterion(outputs_model, targets)
            total_loss += loss.item()

            y_true.extend(targets.cpu().detach().numpy())
            y_pred.extend(outputs_model.cpu().detach().numpy())
        avg_loss = total_loss / len(eval_loader)

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    metrics = {
        'loss': avg_loss,
        'r2': r2,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'pearsonr': pearson_corr,
        'spearmanr': spearman_corr}

    return metrics

def return_values(model, eval_loader, criterion, device):
    model.eval()
    y_true, y_pred= [], []
    with torch.no_grad():
        for batch in eval_loader:
            batch_protein_seqs, batch_lengths, batch_masks, batch_chem_graphs, targets, uniprot_ids_tensor, indicies = [item.to(device) for item in batch]

            outputs_model = model(batch_protein_seqs, batch_chem_graphs, batch_masks)
            y_true.extend(targets.cpu().detach().numpy())
            y_pred.extend(outputs_model.cpu().detach().numpy())

    return y_true, y_pred
