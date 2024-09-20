import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
binding_affinity_module_dir = os.path.join(parentdir, 'BindingAffinityModule')
sys.path.append(binding_affinity_module_dir)


from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch_geometric
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import torch

# my code
from BindingAffinityModule.utils import random_scaffold_split, random_pfam_split
from BindingAffinityModule.dataset import BindingDataset, collate_fn
from BindingAffinityModule.model import BindingModel



def generate_and_save_embeddings(model, loader, df, set_name, device):
    if not os.path.exists('../embeddings'):
        os.makedirs('../embeddings')

    attention_file_path = f"../embeddings/residual_attention_{set_name}_.csv"
    batch_size = loader.batch_size

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if(i % 30 == 0):
                print('Processing Batch: ', i)

            batch_protein_seqs, _, batch_masks, batch_chem_graphs, targets, _ = [item.to(device) for item in batch]
            outputs, combined = model(batch_protein_seqs, batch_chem_graphs, batch_masks)

            combined_np = combined.cpu().numpy()
            predictions_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()

            combined_cols = [f'combined_{j}' for j in range(combined_np.shape[1])]

            attention_df = pd.DataFrame(combined_np, columns=combined_cols)
            attention_df['y_true'] = targets_np
            attention_df['y_pred'] = predictions_np

            start_idx = i * batch_size
            end_idx = start_idx + len(attention_df)
            attention_df['SMILES'] = df['SMILES'][start_idx:end_idx].values
            attention_df['uniprot|pfam'] = df['uniprot|pfam'][start_idx:end_idx].values
            mode = 'w' if i == 0 else 'a'
            header = True if i == 0 else False
            attention_df.to_csv(attention_file_path, mode=mode, header=header, index=False)


def main(args):

    train_data = pd.read_csv(f'../datasets/{args.data_split}/train_rs_{args.seed}_{args.data_split}.csv')
    valid_data = pd.read_csv(f'../datasets/{args.data_split}/valid_rs_{args.seed}_{args.data_split}.csv')
    test_data = pd.read_csv(f'../datasets/{args.data_split}/test_rs_{args.seed}_{args.data_split}.csv')

    with open('../mapping/pfam2value_map.pkl', 'rb') as f:
        uniprot2intmap = pickle.load(f)

    train_loader = DataLoader(BindingDataset(train_data, args.max_length, True), batch_size=args.batch_size_extractor, shuffle=False, collate_fn=lambda batch: collate_fn(batch, args.max_length, uniprot2intmap))
    valid_loader = DataLoader(BindingDataset(valid_data, args.max_length, True), batch_size=args.batch_size_extractor, shuffle=False, collate_fn=lambda batch: collate_fn(batch, args.max_length, uniprot2intmap))
    test_loader = DataLoader(BindingDataset(test_data, args.max_length, True), batch_size=args.batch_size_extractor, shuffle=False, collate_fn=lambda batch: collate_fn(batch, args.max_length, uniprot2intmap))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = BindingModel(num_layer=args.num_layers,
                         emb_dim=args.chem_embed_dim,
                         dropout=args.dropout,
                         gnn_type=args.gnn_type,
                         combined_dim = args.combined_dim).to(device)

    pretrained_weights = torch.load(args.pretrained_model_path, map_location=device)
    model.load_state_dict(pretrained_weights)
    print('Model ready for Embedding Generation......')
    model.eval()

    print('Performing Embedding Generation on Training Set')
    generate_and_save_embeddings(model, train_loader, train_data, f'train_{args.data_split}_seed_{args.seed}', device)
    print('Performing Embedding Generation on Testing Set')
    generate_and_save_embeddings(model, test_loader, test_data, f'test_{args.data_split}_seed_{args.seed}', device)
    print('Performing Embedding Generation on Validation Set')
    generate_and_save_embeddings(model, valid_loader, valid_data, f'valid_{args.data_split}_seed_{args.seed}', device)
    print('Embedding Generation Completed........')


if __name__ == '__main__':
    print('Training Starting........')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, default='./', help='path to pretrained model')
    parser.add_argument('--data_split', type=str, default='random', help='select from [random, scaffold, pfam]')
    parser.add_argument('--seed', type=int, default=42, help='Select Seed, keep the seed same as training to ensure proper embedding extraction')
    parser.add_argument('--max_length', type=int, default=700, help='Maximum Protein Length')
    parser.add_argument('--batch_size_extractor', type=int, default=1024, help='Batch size for training (default: 32)')
    # model parameters
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for GNN')
    parser.add_argument('--chem_embed_dim', type=int, default=300, help='chemical embedding dimension')
    parser.add_argument('--combined_dim', type=int, default=1004, help='chemical embedding dimension')
    parser.add_argument('--gnn_type', type=str, default='gin', help='GNN type')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability for model')

    args = parser.parse_args()

    main(args)
