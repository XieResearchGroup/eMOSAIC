import os
import time
import torch
import pickle
import pandas as pd
from rdkit import Chem
import torch_geometric
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler

from models.ligand_graph_features import *
from models.model_Yang import *
from utils import pad_or_truncate_tensor

class BindingDataset(Dataset):
    """
    Pads sequences till the max length, for the resnet model
    """
    def __init__(self, df, max_length, new_emb_flag):
        self.df = df
        self.protein_embed_dict = self.load_protein_embeddings2() if new_emb_flag else self.load_protein_embeddings()
        self.max_protein_length = max_length
        self.uniprot2intmap = self.load_uniprot2int_map()

        print('Maximum Length of Protein Seqeunce from the Dataset', max_length)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        uniprot_id = self.df.iloc[index]['uniprot|pfam']
        unip_id = self.df.iloc[index]['uniprot|pfam'].split('|')[1]
        protein_embedding = self.protein_embed_dict[uniprot_id]
        chem_smiles = self.df.iloc[index]['SMILES']
        target = torch.tensor(self.df.iloc[index]['pKi'], dtype=torch.float)
        indicies = torch.tensor(index)
        return protein_embedding, chem_smiles, target, unip_id, indicies

    def load_protein_embeddings(self):
        dir_path = '../../../../repo/esm/trial/results_trial/'
        protein_embed_dict = {}

        for file_name in os.listdir(dir_path):
            if file_name.endswith('.pt'):
                file_path = os.path.join(dir_path, file_name)
                data = torch.load(file_path)
                final_rep = data['representations'][33]
                protein_embed_dict[file_name[:-3]] = final_rep

        return protein_embed_dict

    def load_protein_embeddings2(self):
        """
        For the ESM-2 & ESMFold generated embeddings
        """
        protein_embed_dict = {}


        dir_path = '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings/'
        for file_name in os.listdir(dir_path):
            if file_name.endswith('_seq_emb_2.pt'):
                file_path = os.path.join(dir_path, file_name)
                data = torch.load(file_path)
                final_rep = data[0]
                protein_embed_dict[file_name.split('_')[0]] = final_rep

        dir_path = '/home/raid/home/amitesh/EMBBEDINGS/generated_missing_embeddings_1006/'
        for file_name in os.listdir(dir_path):
            if file_name.endswith('_seq_emb_2.pt'):
                file_path = os.path.join(dir_path, file_name)
                data = torch.load(file_path)
                final_rep = data[0]
                protein_embed_dict[file_name.split('_')[0]] = final_rep

        return protein_embed_dict

    def load_uniprot2int_map(self):
        """
        Loading dict for mapping
        """
        dir_path = '../mapping/pfam2value_map.pkl'
        with open(dir_path, 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict

def collate_fn(batch, max_length, uniprot2intmap):
    protein_embeddings, chem_smiles, targets, uniprot_ids, indicies = zip(*batch)
    chem_smiles_list = list(chem_smiles)
    protein_lengths = torch.tensor([p.shape[0] for p in protein_embeddings], dtype=torch.long)
    uniprot_ids_list = list(uniprot_ids)
    mapped_values = [uniprot2intmap.get(identifier, -1) for identifier in uniprot_ids_list]
    mapped_values_list = [torch.tensor(i) for i in mapped_values]

    padded_protein_embeddings = []
    mask = []
    for emb in protein_embeddings:
        length = emb.shape[0]
        if length < max_length:
            padding_length = max_length - length
            padding_tensor = torch.zeros((padding_length, emb.shape[1]))
            padded_emb = torch.cat((emb, padding_tensor), dim=0)
            padded_mask = torch.cat((torch.ones((length, emb.shape[1])), torch.zeros((padding_length, emb.shape[1]))), dim=0)
        else:
            padded_emb = emb[:max_length, :]
            padded_mask = torch.ones((max_length, emb.shape[1]))
        padded_protein_embeddings.append(padded_emb)
        mask.append(padded_mask)

    protein_padded = torch.stack(padded_protein_embeddings)
    mask = torch.stack(mask)
    uniprot_ids_tensor = torch.stack(mapped_values_list)
    targets = torch.stack(targets)
    indicies = torch.stack(indicies)

    chem_graph_list = [mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smiles)) for smiles in chem_smiles]
    chem_graphs_loader = torch_geometric.loader.DataLoader(chem_graph_list, batch_size=len(chem_graph_list), shuffle=False)
    for batch in chem_graphs_loader:
        chem_graphs = batch
        break

    return protein_padded, protein_lengths, mask, chem_graphs, targets, uniprot_ids_tensor, indicies
