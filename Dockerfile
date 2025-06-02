FROM bentoml/model-server:0.11.0-py38
MAINTAINER ersilia

RUN pip install rdkit==2024.3.5
RUN pip install numpy==1.23.1
RUN pip install fastai==1.0.61
RUN pip install scikit-learn==1.3.2
RUN conda install -c conda-forge spacy==3.7.6


WORKDIR /repo
COPY . /repo
