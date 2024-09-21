import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
import os
import yaml
import joblib
import pickle
import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
BINDING_AFFINITY_MODULE_DIR = CURRENT_DIR / 'BindingAffinityModule'
ANOMALY_DETECTION_DIR = CURRENT_DIR / 'AnomalyDetection'

sys.path.append(str(BINDING_AFFINITY_MODULE_DIR))
sys.path.append(str(ANOMALY_DETECTION_DIR))

from BindingAffinityModule.dataset import BindingDataset, collate_fn
from BindingAffinityModule.model import BindingModel2, BindingModel
from BindingAffinityModule.models.ligand_graph_features import mol_to_graph_data_obj_simple
from AnomalyDetection.utils import *

def load_protein_embeddings(uniprot_ids, embedding_dirs):
    """
    Load the protein embeddings for the given list of UniProt IDs.
    """
    protein_embed_dict = {}

    for dir_path in embedding_dirs:
        for file_name in os.listdir(dir_path):
            if file_name.endswith('_seq_emb_2.pt'):
                file_path = os.path.join(dir_path, file_name)
                data = torch.load(file_path)
                final_rep = data[0]
                key = (file_name.split('_')[0]).split('|')[0]
                protein_embed_dict[key] = final_rep

    protein_embeddings = []
    missing_ids = []
    for uniprot_id in uniprot_ids:
        if uniprot_id in protein_embed_dict:
            protein_embeddings.append(protein_embed_dict[uniprot_id])
        else:
            missing_ids.append(uniprot_id)
    if missing_ids:
        raise ValueError(f"Embeddings not found for UniProt IDs: {', '.join(missing_ids)}")
    return protein_embeddings

def prepare_protein_embeddings(protein_embeddings, max_length):
    """
    Pad or truncate the protein embeddings to the required maximum length.
    """
    padded_embeddings = []
    masks = []
    for embedding in protein_embeddings:
        length = embedding.shape[0]
        if length < max_length:
            padding_length = max_length - length
            padding_tensor = torch.zeros((padding_length, embedding.shape[1]))
            padded_embedding = torch.cat((embedding, padding_tensor), dim=0)
            mask = torch.cat((torch.ones((length, embedding.shape[1])), torch.zeros((padding_length, embedding.shape[1]))), dim=0)
        else:
            padded_embedding = embedding[:max_length, :]
            mask = torch.ones((max_length, embedding.shape[1]))
        padded_embeddings.append(padded_embedding)
        masks.append(mask)

    protein_padded = torch.stack(padded_embeddings)
    masks = torch.stack(masks)
    return protein_padded, masks

def prepare_chemical_graphs(smiles_list):
    """
    Convert a list of SMILES strings to a list of graph data objects.
    """
    chem_graphs = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        chem_graph = mol_to_graph_data_obj_simple(mol)
        chem_graphs.append(chem_graph)
    return chem_graphs

def load_binding_model(config, device):
    """
    Initialize the BindingModel and load the trained weights.
    """
    model = BindingModel(
        num_layer=config['num_layers'],
        emb_dim=config['chem_embed_dim'],
        dropout=config['dropout'],
        gnn_type=config['gnn_type'],
        combined_dim=config['combined_dim']
    ).to(device)

    checkpoint_dir = config['checkpoint_dir']
    model_path = os.path.join(checkpoint_dir, 'model_full.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained BindingModel2 weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def load_uncertainty_models(config, device):
    """
    Load the KMeans model, inverse covariance matrices, scaler, and trained MLP model.
    """
    # Load KMeans model
    kmeans_model_path = os.path.join(config['anomaly_detection_dir'], config['kmeans_model_filename'])
    if not os.path.exists(kmeans_model_path):
        raise FileNotFoundError(f"KMeans model not found at {kmeans_model_path}")
    kmeans = joblib.load(kmeans_model_path)

    # Load inverse covariance matrices
    inverse_cov_path = os.path.join(config['anomaly_detection_dir'], config['inverse_cov_matrices_filename'])
    if not os.path.exists(inverse_cov_path):
        raise FileNotFoundError(f"Inverse covariance matrices not found at {inverse_cov_path}")
    with open(inverse_cov_path, 'rb') as f:
        inverse_cov_matrices = pickle.load(f)

    # Load StandardScaler
    scaler_path = os.path.join(config['anomaly_detection_dir'], config['scaler_filename'])
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load trained MLP model
    mlp_model_path = os.path.join(config['anomaly_detection_dir'], config['mlp_model_filename'])
    if not os.path.exists(mlp_model_path):
        raise FileNotFoundError(f"MLP model not found at {mlp_model_path}")
    mlp_model = SimpleMLP(input_size=config['num_clusters']).to(device)
    mlp_model.load_state_dict(torch.load(mlp_model_path, map_location=device))
    mlp_model.eval()

    return kmeans, inverse_cov_matrices, scaler, mlp_model

def compute_mahalanobis_distances(embeddings, centroids, inverse_cov_matrices, labels, num_clusters):
    """
    Compute Mahalanobis distances for the embeddings.
    """
    md_features = np.zeros((embeddings.shape[0], num_clusters))

    for i in range(embeddings.shape[0]):
        cluster_idx = labels[i]
        centroid = centroids[cluster_idx]
        inv_cov = inverse_cov_matrices[cluster_idx]
        diff = embeddings[i] - centroid
        md = np.sqrt(np.dot(np.dot(diff.T, inv_cov), diff))
        md_features[i, cluster_idx] = md

    return md_features

def predict_pKi_with_uncertainty(uniprot_ids, smiles_list, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if len(uniprot_ids) != len(smiles_list):
        raise ValueError("The number of UniProt IDs and SMILES strings must be the same.")

    # Load and prepare protein embeddings
    protein_embeddings = load_protein_embeddings(uniprot_ids, config['embedding_dirs'])
    max_length = config['max_length']
    protein_padded, masks = prepare_protein_embeddings(protein_embeddings, max_length)
    protein_padded = protein_padded.to(device)
    masks = masks.to(device)

    # Prepare chemical graphs
    chem_graphs = prepare_chemical_graphs(smiles_list)
    chem_graph_loader = torch_geometric.loader.DataLoader(chem_graphs, batch_size=len(chem_graphs), shuffle=False)
    for batch in chem_graph_loader:
        chem_graph_batch = batch.to(device)
        break

    # Load the models
    binding_model = load_binding_model(config, device)
    kmeans, inverse_cov_matrices, scaler, mlp_model = load_uncertainty_models(config, device)
    centroids = kmeans.cluster_centers_
    num_clusters = kmeans.n_clusters

    # Generate embeddings and get pKi predictions
    with torch.no_grad():
        # Assuming BindingModel2 returns both pKi and combined embeddings
        output, combined_embeddings = binding_model(protein_padded, chem_graph_batch, masks)
        predicted_pKi = output.cpu().numpy()

        # Ensure combined_embeddings are in float64
        combined_embeddings_pki = combined_embeddings.cpu().numpy().astype(np.float64)

    # Assign clusters to embeddings
    labels = kmeans.predict(combined_embeddings_pki.astype(np.float64))

    # Compute Mahalanobis distances
    md_features = compute_mahalanobis_distances(combined_embeddings_pki, centroids, inverse_cov_matrices, labels, num_clusters)

    # Scale the MD features
    md_features_scaled = scaler.transform(md_features).astype(np.float64)  # Ensure float64

    # Predict uncertainty using the MLP model
    with torch.no_grad():
        md_features_tensor = torch.tensor(md_features_scaled, dtype=torch.float32).to(device)
        uncertainty_predictions = mlp_model(md_features_tensor)
        uncertainty_predictions = uncertainty_predictions.cpu().numpy().squeeze()

    return predicted_pKi, uncertainty_predictions

def main_predict(args):

    uniprot_ids = [uid.strip() for uid in args.uniprot_ids.split(',')]
    smiles_list = [smiles.strip() for smiles in args.smiles_list.split(',')]

    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    config['checkpoint_dir'] = args.checkpoint_dir
    config['dropout'] = args.dropout

    config['embedding_dirs'] = [
        '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings/',
        '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings_1006/'
    ]
    config['anomaly_detection_dir'] = ANOMALY_DETECTION_DIR
    config['kmeans_model_filename'] = f'/home/raid/home/amitesh/ESM/code/v3/MOSAIC/embeddings/datasets/kmeans_model_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.pkl'
    config['inverse_cov_matrices_filename'] = f'/home/raid/home/amitesh/ESM/code/v3/MOSAIC/embeddings/datasets/inverse_cov_matrices_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.pkl'
    config['scaler_filename'] = f'/home/raid/home/amitesh/ESM/code/v3/MOSAIC/embeddings/scaler_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.pkl'
    config['mlp_model_filename'] = '/home/raid/home/amitesh/ESM/code/v3/koff/eMOSAIC-publication/code/results/logs_md/exp21-09-2024-14-34-08/best_mlp_model.pth'
    config['num_clusters'] = args.num_clusters
    config['max_length'] = args.max_length

    try:
        predicted_pKi, uncertainty_predictions = predict_pKi_with_uncertainty(uniprot_ids, smiles_list, config)
        for i in range(len(uniprot_ids)):
            print(f"Protein: {uniprot_ids[i]}, SMILES: {smiles_list[i]}, Predicted pKi: {predicted_pKi[i]:.4f}, Uncertainty: {uncertainty_predictions[i]:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict pKi with uncertainty using the trained models.')
    parser.add_argument('--uniprot_ids', type=str, required=True, help='Comma-separated UniProt IDs of the proteins')
    parser.add_argument('--smiles_list', type=str, required=True, help='Comma-separated SMILES strings of the chemical compounds')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the config YAML file')
    parser.add_argument('--seed', type=int, default=42, help='Select Seed, keep the seed same as training to ensure proper embedding extraction')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations for KMeans')
    parser.add_argument('--scaling', type=str2bool, nargs='?', const=False, default=True, help='Scaling Embeddings for better clustering')
    parser.add_argument('--data_split', type=str, default='random', help='select from [random, scaffold]')
    parser.add_argument('--checkpoint_dir', type=str, required=True, help='Path to the trained BindingModel2 checkpoints')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability used during training')
    parser.add_argument('--max_length', type=int, default=700, help='Maximum protein length used during training')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters for KMeans')

    args = parser.parse_args()
    main_predict(args)
