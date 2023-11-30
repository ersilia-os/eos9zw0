FROM bentoml/model-server:0.11.0-py38
MAINTAINER ersilia

RUN pip install rdkit
RUN pip install numpy==1.23.1
RUN pip install fastai==1.0.61
RUN pip install scikit-learn
RUN pip install spacy


WORKDIR /repo
COPY . /repo
