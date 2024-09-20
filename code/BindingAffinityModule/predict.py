# predict.py

import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from rdkit import Chem
import yaml
import pickle
import argparse
import os
import pandas as pd
import numpy as np

# Custom modules
from dataset import BindingDataset, collate_fn
from model import BindingModel2
from models.ligand_graph_features import mol_to_graph_data_obj_simple

def load_protein_embeddings(uniprot_ids):
    """
    Load the protein embeddings for the given list of UniProt IDs.
    """
    protein_embed_dict = {}

    # Directories containing embeddings
    dir_paths = [
        '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings/',
        '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings_1006/'
    ]

    # Load all embeddings into a dictionary
    for dir_path in dir_paths:
        for file_name in os.listdir(dir_path):
            if file_name.endswith('_seq_emb_2.pt'):
                file_path = os.path.join(dir_path, file_name)
                data = torch.load(file_path)
                final_rep = data[0]
                key = (file_name.split('_')[0]).split('|')[0]
                protein_embed_dict[key] = final_rep

    # Retrieve embeddings for the requested UniProt IDs
    protein_embeddings = []
    for uniprot_id in uniprot_ids:
        if uniprot_id in protein_embed_dict:
            protein_embeddings.append(protein_embed_dict[uniprot_id])
        else:
            raise ValueError(f"Embedding for UniProt ID {uniprot_id} not found.")
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
    # Stack to create batch dimension
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

def load_model(config, device):
    """
    Initialize the model and load the trained weights.
    """
    model = BindingModel2(
        num_layer=config['num_layers'],
        emb_dim=config['chem_embed_dim'],
        dropout=config['dropout'],
        gnn_type=config['gnn_type'],
        combined_dim=config['combined_dim']
    ).to(device)

    checkpoint_dir = config['checkpoint_dir']
    model_path = os.path.join(checkpoint_dir, 'model_full_final.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights not found at {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_pKi(uniprot_ids, smiles_list, config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if len(uniprot_ids) != len(smiles_list):
        raise ValueError("The number of UniProt IDs and SMILES strings must be the same.")

    # Load and prepare protein embeddings
    protein_embeddings = load_protein_embeddings(uniprot_ids)
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

    # Load the model
    model = load_model(config, device)

    # Make predictions
    with torch.no_grad():
        outputs = model(protein_padded, chem_graph_batch, masks)
        predicted_pKi = outputs.cpu().numpy()

    return predicted_pKi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict pKi using the trained model.')
    parser.add_argument('--uniprot_ids', type=str, required=True, help='Comma-separated UniProt IDs of the proteins')
    parser.add_argument('--smiles_list', type=str, required=True, help='Comma-separated SMILES strings of the chemical compounds')
    parser.add_argument('--config_path', type=str, default='config.yaml', help='Path to the config YAML file')
    args = parser.parse_args()

    # Parse the lists
    uniprot_ids = [uid.strip() for uid in args.uniprot_ids.split(',')]
    smiles_list = [smiles.strip() for smiles in args.smiles_list.split(',')]

    # Load configuration
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update config with additional required parameters
    config['checkpoint_dir'] = '../results/logs/exp17-09-2024-22-34-15/'
    config['dropout'] = 0.2

    try:
        predicted_pKi = predict_pKi(uniprot_ids, smiles_list, config)
        # Print the results
        for i in range(len(uniprot_ids)):
            print(f"Protein: {uniprot_ids[i]}, SMILES: {smiles_list[i]}, Predicted pKi: {predicted_pKi[i]:.4f}")
    except Exception as e:
        print(f"Error: {str(e)}")
