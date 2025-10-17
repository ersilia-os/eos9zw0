# Molecular Prediction Model Fine-Tuning (MolPMoFiT) encodings

Using self-supervised learning, the authors pre-trained a large model using one millon unlabelled molecules from ChEMBL. This model can subsequently be fine-tuned for various QSAR tasks. Here, we provide the encodings for the molecular structures using the pre-trained model, not the fine-tuned QSAR models.

This model was incorporated on 2023-11-06.


## Information
### Identifiers
- **Ersilia Identifier:** `eos9zw0`
- **Slug:** `molpmofit`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Descriptor`, `Embedding`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `400`
- **Output Consistency:** `Fixed`
- **Interpretation:** Embedding vectors of each smiles are obtained, represented in a matrix, where each row is a vector of embedding of each smiles character, with a dimension of 400. The pretrained model is loaded using the fastai library

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| dim_000 | float |  | MolPMoFit encoding dimension index 0 |
| dim_001 | float |  | MolPMoFit encoding dimension index 1 |
| dim_002 | float |  | MolPMoFit encoding dimension index 2 |
| dim_003 | float |  | MolPMoFit encoding dimension index 3 |
| dim_004 | float |  | MolPMoFit encoding dimension index 4 |
| dim_005 | float |  | MolPMoFit encoding dimension index 5 |
| dim_006 | float |  | MolPMoFit encoding dimension index 6 |
| dim_007 | float |  | MolPMoFit encoding dimension index 7 |
| dim_008 | float |  | MolPMoFit encoding dimension index 8 |
| dim_009 | float |  | MolPMoFit encoding dimension index 9 |

_10 of 400 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos9zw0](https://hub.docker.com/r/ersiliaos/eos9zw0)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos9zw0.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos9zw0.zip)

### Resource Consumption
- **Model Size (Mb):** `122`
- **Environment Size (Mb):** `6523`
- **Image Size (Mb):** `6594.2`

**Computational Performance (seconds):**
- 10 inputs: `41.35`
- 100 inputs: `104.76`
- 10000 inputs: `1599.82`

### References
- **Source Code**: [https://github.com/XinhaoLi74/MolPMoFiT](https://github.com/XinhaoLi74/MolPMoFiT)
- **Publication**: [https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00430-x](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00430-x)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2020`
- **Ersilia Contributor:** [GemmaTuron](https://github.com/GemmaTuron)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [None](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos9zw0
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos9zw0
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
