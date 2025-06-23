FROM bentoml/model-server:0.11.0-py38
MAINTAINER ersilia

RUN pip install rdkit==2024.3.5
RUN conda install -c conda-forge numpy==1.23.1
RUN pip install fastai==1.0.61
RUN pip install scikit-learn==1.3.2
RUN pip install spacy==3.7.6


WORKDIR /repo
COPY . /repo
