from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.covariance import LedoitWolf
from sklearn.cluster import KMeans
from utils import *
import pandas as pd
import numpy as np
import argparse

def main(args):
    if not os.path.exists('../embeddings/datasets'):
        os.makedirs('../embeddings/datasets')

    train_df = pd.read_csv(f'../embeddings/residual_attention_train_{args.data_split}_seed_{args.seed}_.csv')
    test_df = pd.read_csv(f'../embeddings/residual_attention_test_{args.data_split}_seed_{args.seed}_.csv')
    valid_df = pd.read_csv(f'../embeddings/residual_attention_valid_{args.data_split}_seed_{args.seed}_.csv')
    train_embeddings = train_df[[f'combined_{i}' for i in range(1004)]].values
    test_embeddings = test_df[[f'combined_{i}' for i in range(1004)]].values
    valid_embeddings = valid_df[[f'combined_{i}' for i in range(1004)]].values

    if(args.scaling == True):
        print('Performing Normalization of Embeddings before clustering......')
        scaler = MinMaxScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        test_embeddings = scaler.transform(test_embeddings)
        valid_embeddings = scaler.transform(valid_embeddings)

    clustered_embeddings = {}
    print('KMeans Starting......')
    kmeans = KMeans(n_clusters=args.num_clusters, max_iter=args.iters)
    kmeans.fit(train_embeddings)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    clustered_embeddings = {i: train_embeddings[labels == i] for i in range(args.num_clusters)}
    print('KMeans clustering completed......')

    lw = LedoitWolf()
    cluster_covariances = {i: lw.fit(clustered_embeddings[i]).covariance_ for i in range(0, args.num_clusters)}
    inverse_cov_matrices = {i: np.linalg.inv(cluster_covariances[i]) for i in cluster_covariances}
    print('Cluster Covariances Generated......')

    X_train = mahalanobis_distance_dataset(train_embeddings, centroids, inverse_cov_matrices, args.num_clusters)
    X_test = mahalanobis_distance_dataset(test_embeddings, centroids, inverse_cov_matrices, args.num_clusters)
    X_valid = mahalanobis_distance_dataset(valid_embeddings, centroids, inverse_cov_matrices, args.num_clusters)

    df_X_train = pd.DataFrame(X_train)
    df_X_test = pd.DataFrame(X_test)
    df_X_valid = pd.DataFrame(X_valid)

    df_X_train['cluster_label'] = labels
    df_X_test['cluster_label'] = kmeans.predict(test_embeddings)
    df_X_valid['cluster_label'] = kmeans.predict(valid_embeddings)

    df_X_train['y_true'] = train_df['y_true'].values
    df_X_test['y_true'] = test_df['y_true'].values
    df_X_valid['y_true'] = valid_df['y_true'].values

    df_X_train['y_pred'] = train_df['y_pred'].values
    df_X_test['y_pred'] = test_df['y_pred'].values
    df_X_valid['y_pred'] = valid_df['y_pred'].values

    df_X_train['SMILES'] = train_df['SMILES'].values
    df_X_test['SMILES'] = test_df['SMILES'].values
    df_X_valid['SMILES'] = valid_df['SMILES'].values

    df_X_train['uniprot|pfam'] = train_df['uniprot|pfam'].values
    df_X_test['uniprot|pfam'] = test_df['uniprot|pfam'].values
    df_X_valid['uniprot|pfam'] = valid_df['uniprot|pfam'].values

    print('Dataset Ready for Anomaly Detection')

    df_X_train.to_csv(f'../embeddings/datasets/md_dataset_train_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')
    df_X_test.to_csv(f'../embeddings/datasets/md_dataset_test_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')
    df_X_valid.to_csv(f'../embeddings/datasets/md_dataset_valid_{args.data_split}_num_clusters_{args.num_clusters}_iters_{args.iters}_scaling_{args.scaling}_seed_{args.seed}.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating Datasets')
    parser.add_argument('--data_split', type=str, default='random', help='select from [random, scaffold]')
    parser.add_argument('--num_clusters', type=int, default=50, help='Number of clusters for KMeans')
    parser.add_argument('--iters', type=int, default=10, help='Number of iterations for KMeans')
    parser.add_argument('--scaling', type=str2bool, nargs='?', const=False, default=True, help='Scaling Embeddings for better clustering')
    parser.add_argument('--seed', type=int, default=42, help='Select Seed, keep the seed same as training to ensure proper embedding extraction')

    args = parser.parse_args()
    main(args)
