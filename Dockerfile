FROM bentoml/model-server:0.11.0-py310
MAINTAINER ersilia

RUN conda install -c conda-forge compilers==1.9.0
RUN conda install -c conda-forge libxcrypt==4.4.36

# Install packages in order, with numpy constraints
RUN pip install rdkit==2024.3.5
RUN pip install scikit-learn==1.3.2
RUN pip install spacy==3.8.11
RUN pip install fastai==1.0.61

# Force numpy to compatible version AFTER all other packages
RUN pip install numpy==1.23.5 

WORKDIR /repo
COPY . /repo