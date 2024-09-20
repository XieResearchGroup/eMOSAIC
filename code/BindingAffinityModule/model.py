import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F

from models.model_Yang import GNN
from models.resnet import *

class AttentivePooling(nn.Module):
    """ Improved attentive pooling network """
    def __init__(self, chem_hidden_size, prot_hidden_size):
        super(AttentivePooling, self).__init__()
        self.chem_hidden_size = chem_hidden_size
        self.prot_hidden_size = prot_hidden_size
        self.param = nn.Parameter(torch.zeros(chem_hidden_size, prot_hidden_size))
        self.dropout = nn.Dropout(p=0.4)
        self.relu = nn.ReLU()

    def forward(self, chem_embedding, prot_embedding):
        """ Calculate attentive pooling attention weighted representation and
        attention scores for the two inputs.
        Args:
            chem_embedding: chemical embedding of size (batch_size, chem_hidden_size)
            prot_embedding: protein embedding of size (batch_size, prot_hidden_size)
        Returns:
            rep_chem: attention weighted representations for the chemical embedding
            rep_prot: attention weighted representations for the protein embedding
        """

        param = self.relu(self.param)
        param = self.dropout(param)

        wm_chem = torch.matmul(prot_embedding.unsqueeze(1), param.transpose(0, 1).unsqueeze(0))
        wm_chem = self.relu(wm_chem)
        wm_chem = self.dropout(wm_chem)

        wm_prot = torch.matmul(chem_embedding.unsqueeze(1), param.unsqueeze(0))
        wm_prot = self.relu(wm_prot)
        wm_prot = self.dropout(wm_prot)

        score_chem = F.softmax(wm_chem, dim=2)
        score_prot = F.softmax(wm_prot, dim=2)

        rep_chem = torch.sum(chem_embedding.unsqueeze(1) * score_chem, dim=1)
        rep_prot = torch.sum(prot_embedding.unsqueeze(1) * score_prot, dim=1)

        return rep_chem, rep_prot


class BindingModel2(nn.Module):
    def __init__(self, num_layer, emb_dim, dropout,
                 gnn_type, combined_dim):
        super(BindingModel2, self).__init__()
        self.combined_dim = combined_dim

        self.prot_resnet = ResnetEncoderModel(in_channels = 1, blocks_sizes = [16, 32, 64, 32, 16], depths=[2, 2, 2, 2, 2], activation = 'relu')
        self.chem_gnn = GNN(
            num_layer=num_layer, emb_dim=emb_dim, JK="last", drop_ratio=dropout, gnn_type=gnn_type, pretrained = True
        )
        self.attention = AttentivePooling(chem_hidden_size=emb_dim, prot_hidden_size=704)
        self.mlp = nn.Sequential(
            nn.Linear(self.combined_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, batch_protein_seqs, batch_chem_graphs, batch_masks):

        prot_res_inp = batch_protein_seqs.unsqueeze(dim=1)
        prot_emb = self.prot_resnet(prot_res_inp)
        batch_masks_reshaped = F.interpolate(batch_masks.unsqueeze(dim=1), prot_emb.size()[2:]).to(
            batch_protein_seqs.device)
        threshold = 0.5
        batch_binary_masks = torch.where(batch_masks_reshaped >= threshold,
                                         torch.tensor(1).to(batch_protein_seqs.device),
                                         torch.tensor(0).to(batch_protein_seqs.device))

        prot_emb = prot_emb * batch_binary_masks
        prot_emb = prot_emb.reshape(prot_emb.shape[0], 1, -1).squeeze(1).to(batch_protein_seqs.device)
        node_representation = self.chem_gnn(batch_chem_graphs.x, batch_chem_graphs.edge_index, batch_chem_graphs.edge_attr)
        batch_chem_graphs_repr_masked, mask_graph = torch_geometric.utils.to_dense_batch(node_representation,
                                                                                         batch_chem_graphs.batch)
        batch_chem_graphs_repr_pooled = batch_chem_graphs_repr_masked.sum(axis=1)

        (prot_emb_final, chem_emb_final) = self.attention(batch_chem_graphs_repr_pooled, prot_emb)
        combined = torch.cat((prot_emb_final, chem_emb_final), dim=1)

        regression_out = self.mlp(combined).squeeze()
        return regression_out
