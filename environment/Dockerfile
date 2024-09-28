FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

ENV PATH /root/miniconda3/bin:$PATH

WORKDIR /eMOSIAC

COPY environment.yml /eMOSIAC/environment.yml

RUN conda env create -f environment.yml && \
    echo "source activate eMOSIAC" > ~/.bashrc && \
    /bin/bash -c "source activate eMOSIAC && \
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.1+cu113.html && \
    pip install torch-geometric==2.2.0 && \
    conda install -c conda-forge rdkit"


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "eMOSIAC", "/bin/bash"]
