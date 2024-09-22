import torch.nn.functional as F
from datetime import datetime
from scipy import spatial
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import os

def mahalanobis_distance_dataset(embeddings, centroids, inverse_cov_matrices, num_clusters):
    num_clusters = len(centroids)
    distances = []
    for point in embeddings:
        point_distances = []
        for i in range(num_clusters):
            centroid = centroids[i]
            distance = spatial.distance.mahalanobis(point, centroid, inverse_cov_matrices[i])
            point_distances.append(distance)
        distances.append(point_distances)
    return (np.array(distances)).reshape(embeddings.shape[0], -1)

def set_up_exp_folder(path):
    now = datetime.now()
    timestamp = now.strftime("%d-%m-%Y-%H-%M-%S")
    print('timestamp: ',timestamp)
    save_folder = path
    if os.path.exists(save_folder) == False:
            os.mkdir(save_folder)
    checkpoint_dir = '{}/exp{}/'.format(save_folder, timestamp)
    if os.path.exists(checkpoint_dir ) == False:
            os.mkdir(checkpoint_dir )
    return checkpoint_dir

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def load_and_split_data(data, args):
    md_columns = [f"{i}" for i in range(args.num_clusters)]

    X = data[md_columns].values
    if (args.include_y_pred_flag == 'True'):
        X_nn = data['y_pred'].values.reshape(-1, 1)
        X = np.concatenate((X, X_nn), axis=1)

    y = abs(data['y_true'] - data['y_pred'])
    return X, y

def pad_or_truncate_tensor(tensor, threshold = 500):
    """
    function to pad or truncate the tensors
    """
    x = tensor.size(0)
    if x < threshold:
        padding = torch.zeros((threshold - x, 1280))
        padded_tensor = torch.cat((tensor, padding), dim=0)
        return padded_tensor
    elif x > threshold:
        truncated_tensor = tensor[:threshold]
        return truncated_tensor
    else:
        return tensor

class SimpleMLP(nn.Module):
    def __init__(self, input_size=50):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    def forward(self, x):
        x1 = F.relu(self.fc1(x))
        x2 = F.relu(self.fc2(x1))
        x2 = self.fc3(x2)
        return x2

def evaluate_model(model, valid_loader, loss_fn, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item() * data.size(0)
    avg_loss = total_loss / len(valid_loader.dataset)
    return avg_loss

def create_df_results(pred, df, file_name, checkpoint_dir):
    df_results = pd.DataFrame({
        'predicted_residue': pred,
        'true_residue': abs(df['y_true'] - df['y_pred']),
        'y_true': df['y_true'],
        'y_pred': df['y_pred'],
        'SMILES': df['SMILES'],
        'uniprot|pfam': df['uniprot|pfam']
    })
    tolerance = 0.5
    df_results[f'Tolerance {tolerance}'] = np.where(df_results['predicted_residue'] < tolerance, 'Normal', 'Outlier')
    file_path = os.path.join(checkpoint_dir, f'residues_values_analysis_{file_name}.csv')
    df_results.to_csv(file_path)
