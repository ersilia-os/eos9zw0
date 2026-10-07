# imports
import pickle
from collections import defaultdict
from rdkit import Chem, RDLogger
import csv
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages
from fastai.text.models.awd_lstm import AWD_LSTM
from utils import MolTokenizer, special_tokens
import torch

import sys
import os

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
path_to_checkpoint = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
path_vocab = os.path.join(path_to_checkpoint, "ChemBL_atom_vocab.pkl")
path_encoder = os.path.join(path_to_checkpoint, "models", "ChemBL_atom_encoder.pth")

EMB_SZ = 400 # size of the final LSTM layer output
BATCH_SIZE = 128

#Read the vocabulary; tokens missing from it map to index 0, as in fastai's numericalisation
with open(path_vocab, 'rb') as f:
    itos = pickle.load(f)
stoi = {}
for i, token in enumerate(itos):
    stoi.setdefault(token, i)

tokenizer = MolTokenizer(special_tokens=special_tokens)

#Load the pretrained MolPMoFiT encoder (AWD-LSTM, 3 layers)
encoder = AWD_LSTM(vocab_sz=len(itos), emb_sz=EMB_SZ, n_hid=1152, n_layers=3, pad_token=1)
encoder.load_state_dict(torch.load(path_encoder, map_location="cpu"))
encoder.eval()

def tokens_to_ids(smiles):
    return [stoi.get(token, 0) for token in tokenizer.tokenizer(smiles)]

# my model
@torch.no_grad()
def my_model(smiles_list):
    """ULMFiT concat pooling of the final LSTM layer: [last hidden state, max-pool, mean-pool], 1200 features.

    Molecules are batched by token length so that no padding is needed: each molecule gets the same
    vector it would get on its own. Molecules RDKit cannot parse are returned as None.
    """
    outputs = [None] * len(smiles_list)
    by_length = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        if Chem.MolFromSmiles(smiles) is None:
            continue
        by_length[len(tokens_to_ids(smiles))].append(i)
    for indices in by_length.values():
        for start in range(0, len(indices), BATCH_SIZE):
            batch = indices[start:start + BATCH_SIZE]
            x = torch.tensor([tokens_to_ids(smiles_list[i]) for i in batch], dtype=torch.long)
            encoder.reset() # the encoder keeps its hidden state between calls
            _, layer_outputs = encoder(x)
            last_layer = layer_outputs[-1]
            pooled = torch.cat([last_layer[:, -1], last_layer.max(dim=1)[0], last_layer.mean(dim=1)], dim=1)
            for j, i in enumerate(batch):
                outputs[i] = list(pooled[j].numpy()) # float32 values, written in their shortest form
    return outputs

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

# write output in a .csv file; molecules that could not be processed are written as empty cells
n_features = 3 * EMB_SZ
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["feat_{0}".format(str(i).zfill(4)) for i in range(n_features)])  # header
    for o in outputs:
        writer.writerow(o if o is not None else [None] * n_features)
