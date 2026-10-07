# Molecular Prediction Model Fine-Tuning (MolPMoFiT) encodings

Represents a molecule as 1,200 features drawn from MolPMoFiT, an AWD-LSTM language model that Li and Fourches pretrained on one million unlabelled ChEMBL structures by adapting the ULMFiT inductive transfer learning recipe from natural language, then fine-tuned for lipophilicity, solvation, HIV activity and blood-brain barrier penetration. Only the pretrained encoder is served, not any fine-tuned endpoint; its last LSTM layer is summarised by ULMFiT concat pooling (last hidden state, max-pool and mean-pool, 400 features each), the input its prediction heads use.

This model was incorporated on 2023-11-06.Last packaged on 2026-10-07.

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
- **Output Dimension:** `1200`
- **Output Consistency:** `Fixed`
- **Interpretation:** Last hidden state, max-pool and mean-pool of the final LSTM layer of a ChEMBL-pretrained SMILES language model, 400 features each.

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| feat_0000 | float |  | MolPMoFiT last hidden state (dimension 0) |
| feat_0001 | float |  | MolPMoFiT last hidden state (dimension 1) |
| feat_0002 | float |  | MolPMoFiT last hidden state (dimension 2) |
| feat_0003 | float |  | MolPMoFiT last hidden state (dimension 3) |
| feat_0004 | float |  | MolPMoFiT last hidden state (dimension 4) |
| feat_0005 | float |  | MolPMoFiT last hidden state (dimension 5) |
| feat_0006 | float |  | MolPMoFiT last hidden state (dimension 6) |
| feat_0007 | float |  | MolPMoFiT last hidden state (dimension 7) |
| feat_0008 | float |  | MolPMoFiT last hidden state (dimension 8) |
| feat_0009 | float |  | MolPMoFiT last hidden state (dimension 9) |

_10 of 1200 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos9zw0](https://hub.docker.com/r/ersiliaos/eos9zw0)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos9zw0.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos9zw0.zip)

### Resource Consumption
- **Model Size (Mb):** `122`
- **Environment Size (Mb):** `7262`
- **Image Size (Mb):** `7501.1`

**Computational Performance (seconds):**
- 10 inputs: `29.68`
- 100 inputs: `23.32`
- 10000 inputs: `296.57`

### References
- **Source Code**: [https://github.com/XinhaoLi74/MolPMoFiT](https://github.com/XinhaoLi74/MolPMoFiT)
- **Publication**: [https://doi.org/10.1186/s13321-020-00430-x](https://doi.org/10.1186/s13321-020-00430-x)
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
