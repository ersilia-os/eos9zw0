# Molecular Prediction Model Fine-Tuning (MolPMoFiT)

Using self-supervised learning, the authors pre-trained a large model using one millon unlabelled molecules from ChEMBL. This model can subsequently be fine-tuned for various QSAR tasks. Here, we provide the encodings for the molecular structures using the pre-trained model, not the fine-tuned QSAR models.

## Identifiers

* EOS model ID: `eos9zw0`
* Slug: `molpmofit`

## Characteristics

* Input: `Compound`
* Input Shape: `Single`
* Task: `Representation`
* Output: `Other value`
* Output Type: `Float`
* Output Shape: `Matrix`
* Interpretation: Embedding vectors of each smiles are obtained, represented in a matrix, where each row is a vector of embedding of each smiles character, with a dimension of 400. The pretrained model is loaded using the fastai library

## References

* [Publication](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00430-x)
* [Source Code](https://github.com/XinhaoLi74/MolPMoFiT)
* Ersilia contributor: [GemmaTuron](https://github.com/GemmaTuron)

## Ersilia model URLs
* [GitHub](https://github.com/ersilia-os/eos9zw0)
* [AWS S3](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos9zw0.zip)
* [DockerHub](https://hub.docker.com/r/ersiliaos/eos9zw0) (AMD64, ARM64)

## Citation

If you use this model, please cite the [original authors](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-020-00430-x) of the model and the [Ersilia Model Hub](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff).

## License

This package is licensed under a GPL-3.0 license. The model contained within this package is licensed under a CC license.

Notice: Ersilia grants access to these models 'as is' provided by the original authors, please refer to the original code repository and/or publication if you use the model in your research.

## About Us

The [Ersilia Open Source Initiative](https://ersilia.io) is a Non Profit Organization ([1192266](https://register-of-charities.charitycommission.gov.uk/charity-search/-/charity-details/5170657/full-print)) with the mission is to equip labs, universities and clinics in LMIC with AI/ML tools for infectious disease research.

[Help us](https://www.ersilia.io/donate) achieve our mission!