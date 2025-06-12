# imports
import pickle
from rdkit import RDLogger 
import pandas as pd
RDLogger.DisableLog('rdApp.*') # switch off RDKit warning messages
from fastai import *
from fastai.text import *
from utils import *
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

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
tok = Tokenizer(partial(MolTokenizer, special_tokens = special_tokens), n_cpus=6, pre_rules=[], post_rules=[])

#read the dataset for  QSAR tasks. 
#Load the data y data augmentation
bbbp = pd.read_csv(path_bbp)

train, test = train_test_split(bbbp,
    test_size=0.1, shuffle = True, random_state = 8)

train, val = train_test_split(train,
    test_size=0.1, shuffle = True, random_state = 42)

bs = 128 #batch size

#Create the specific group of data for the language model, the same data used for train the model.
#Build the fastai databunch
qsar_db = TextClasDataBunch.from_df(path_to_checkpoint, train, val, bs=bs, tokenizer=tok, 
                                    chunksize=50000, text_cols='smiles',label_cols='p_np', 
                                    vocab=vocab, max_vocab=60000, include_bos=False)

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
    tokenizer = MolTokenizer()
    for smile in smiles_list:
        smile_tokenizer=tokenizer.tokenizer(smile)
        indices = [qsar_db.vocab.itos.index(token) for token in smile_tokenizer]
        embes=embs_v1[indices][:, :].numpy()
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


