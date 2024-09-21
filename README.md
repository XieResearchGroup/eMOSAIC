## eMOSAIC: Multi-modal Out-of-distribution Uncertainty Quantification Streamlines Large-scale Polypharmacology

Contains the code for training, testing our proposed eMOSAIC model for prediction of binding affinity and the uncertainty associated with it.

## Dataset

eMOSAIC requires the following for training:

1. A csv file containing the protein (with it's UniProt ID and Pfam family), SMILES, pKi value (binding affinity). 
2. A directory of ESM-2 and/or ESMFold generated embeddings.

The dataset folder contains more information and scripts to help generate the embeddings and create the dataset.

## Environment

1. CUDA : x Python : y
2. The other dependencies can be installed with the requirements.txt file present under the environment folder

## Usage

1. We train the binding affinity prediction model present in code/BindingAffinityModule/ folder using the following:

```python
python main.py
```

2. Once we have the trained binding affinity prediction model, we train eMOSAIC for uncertainty quantification (from code/AnomalyDetection/), this extracts the embeddings, clusters them and then learns the residue for accurate uncertainty quantification:
```python
python main.py
```

3. For using pretrained model, use the following (from code/BindingAffinityModule/):

```python
python predict.py --smiles_list "Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, COC(=O)c1cccc(COc2ccc3[nH]c(SCC(=O)c4ccc(O)c(O)c4)nc3c2)c1" --uniprot_ids "Q07817, Q07820, P47871"
```

4. For uncertainty quantification, as well pKi prediction, we can use the predict_pKi_uncertainty.py file:

```python
python predict_pki_uncertainty.py --smiles_list "Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, Cc1cc(Oc2ccc(/C=C3\\SC(=O)N([C@@H](Cc4ccccc4)C(=O)O)C3=O)cc2)cc(C)c1Cl, COC(=O)c1cccc(COc2ccc3[nH]c(SCC(=O)c4ccc(O)c(O)c4)nc3c2)c1" --uniprot_ids "Q07817, Q07820, P47871" --data_split=scaffold --num_clusters=50 --iters=10 --scaling=True --seed=42 --checkpoint_dir="/results/logs/exp08-02-2024-05-02-20/"
```


