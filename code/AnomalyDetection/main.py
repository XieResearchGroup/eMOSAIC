from embedding_generator import main as embedding_generator_main
from dataset_generator import main as dataset_generator_main
from anomaly_detection import main as anomaly_detection_main
from utils import str2bool
import argparse


def main(args):
    embedding_generator_main(args)
    dataset_generator_main(args)
    anomaly_detection_main(args)

    print('Process Completed!')
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eMOSIAC')
    parser.add_argument('--pretrained_model_path', type=str, default='./', help='path to pretrained model')
    parser.add_argument('--data_split', type=str, default='random', help='select from [random, scaffold]')
    parser.add_argument('--seed', type=int, default=42, help='Select Seed, keep the seed same as training to ensure proper embedding extraction')
    parser.add_argument('--batch_size_extractor', type=int, default=1024, help='Batch size for training (default: 32)')
    parser.add_argument('--result_path', type=str, default='../results/logs_md/', help='path to save best model')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters for KMeans')
    parser.add_argument('--model_type', type=str, default='xgb', help='Type of the model: rf, xgb, lr, or mlp')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training for MLP')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--iters', type=int, default=50, help='Number of iterations for KMeans')
    parser.add_argument('--scaling', type=str2bool, nargs='?', const=False, default=True, help='Scaling Embeddings for better clustering')
    parser.add_argument('--max_length', type=int, default=700, help='Maximum Protein Length')
    # model parameters
    parser.add_argument('--num_layers', type=int, default=2, help='number of layers for GNN')
    parser.add_argument('--chem_embed_dim', type=int, default=300, help='chemical embedding dimension')
    parser.add_argument('--combined_dim', type=int, default=1004, help='chemical embedding dimension')
    parser.add_argument('--gnn_type', type=str, default='gin', help='GNN type')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout probability for model')

    args = parser.parse_args()
    main(args)
