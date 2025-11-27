# imports
import pickle
from rdkit import RDLogger 
import pandas as pd
import numpy as np
import csv
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages
from fastai import *
from fastai.text import *
from utils import *
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
import torch

import sys
import os

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
path= os.path.abspath(os.path.join(root,".."))
path_to_checkpoint=os.path.abspath(os.path.join(root,"..", "..", "checkpoints"))
path_vocab= os.path.abspath(os.path.join(root,"..","..","checkpoints", "ChemBL_atom_vocab.pkl"))
path_bbp= os.path.abspath(os.path.join(root,"..","data","QSAR","bbbp.csv"))

#Read the vocabulary
with open(f'{path_vocab}', 'rb') as f:
    orig_itos = pickle.load(f)

#Load the vocabulary
vocab = Vocab(orig_itos)

#Initialize the Tokenizer
tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus=1, pre_rules=[], post_rules=[])

#read the dataset for  QSAR tasks. 
#Load the data y data augmentation
bbbp = pd.read_csv(path_bbp)

train, test = train_test_split(bbbp,
    test_size=0.1, shuffle = True, random_state = 8)

train, val = train_test_split(train,
    test_size=0.1, shuffle = True, random_state = 42)

bs = 128 #batch size

# Debug prints
print(f"Train shape: {train.shape}")
print(f"Val shape: {val.shape}")
print(f"Columns: {train.columns.tolist()}")
print(f"Sample SMILES: {train['smiles'].iloc[0]}")
print(f"Sample label: {train['p_np'].iloc[0]}")

# Manually tokenize the data first
print("Tokenizing data...")
tokenizer_func = MolTokenizer(special_tokens=special_tokens)

def tokenize_smiles(smiles):
    return tokenizer_func.tokenizer(smiles)

def numericalize_tokens(tokens, vocab):
    return [vocab.stoi.get(token, vocab.stoi.get('xxunk', 0)) for token in tokens]

# Tokenize and numericalize training data
train_tokens = [tokenize_smiles(s) for s in train['smiles'].values]
train_numerics = [numericalize_tokens(t, vocab) for t in train_tokens]
train_labels = train['p_np'].values

# Tokenize and numericalize validation data
val_tokens = [tokenize_smiles(s) for s in val['smiles'].values]
val_numerics = [numericalize_tokens(t, vocab) for t in val_tokens]
val_labels = val['p_np'].values

# Pad sequences to the same length
def pad_sequences(sequences, pad_value=1, max_len=None):
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + [pad_value] * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return padded

print("Padding sequences...")
train_padded = pad_sequences(train_numerics)
val_padded = pad_sequences(val_numerics, max_len=len(train_padded[0]))

# Convert to tensors
train_x = torch.tensor(train_padded, dtype=torch.long)
train_y = torch.tensor(train_labels, dtype=torch.long)
val_x = torch.tensor(val_padded, dtype=torch.long)
val_y = torch.tensor(val_labels, dtype=torch.long)

# Create custom dataset class with required attributes
class TextDataset(torch.utils.data.TensorDataset):
    def __init__(self, x, y, c):
        super().__init__(x, y)
        self.c = c  # number of classes

# Get number of classes
num_classes = len(np.unique(train_labels))

# Create datasets with c attribute
train_ds = TextDataset(train_x, train_y, num_classes)
val_ds = TextDataset(val_x, val_y, num_classes)

# Create DataLoaders
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False)

# Create databunch manually
qsar_db = DataBunch(train_dl, val_dl, path=path_to_checkpoint)
qsar_db.vocab = vocab
qsar_db.c = num_classes

#create the classification/regression learner.
cls_learner = text_classifier_learner(qsar_db, AWD_LSTM, pretrained=False, drop_mult=0.1)

#The encoder of the model is loaded before training, and then access the first layer, the embeddings layer, and obtain them
#learner.load_encoder() will load the model from path/models/
cls_learner.load_encoder('ChemBL_atom_encoder')

def get_normalized_embeddings():
    return F.normalize(cls_learner.model[0].module.encoder.weight)

embs_v1 = get_normalized_embeddings()

# my model
def my_model(smiles_list):
    list_embeddings=[]
    tokenizer = MolTokenizer(special_tokens=special_tokens)
    for smile in smiles_list:
        smile_tokenizer=tokenizer.tokenizer(smile)
        indices = [vocab.stoi.get(token, vocab.stoi.get('xxunk', 0)) for token in smile_tokenizer]
        if len(indices) == 0:
            # Handle empty tokenization - use zero vector
            list_embeddings.append(np.zeros(400))
            continue
        embes=embs_v1[indices][:, :].detach().cpu().numpy()
        embes = np.mean(embes, axis=0)
        list_embeddings.append(embes)

    return list_embeddings

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    smiles_list = [r[0] for r in reader]

# run model
outputs = my_model(smiles_list)

#check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["dim_{0}".format(str(i).zfill(3)) for i in range(400)])  # header
    for o in outputs:
        writer.writerow(list(o))