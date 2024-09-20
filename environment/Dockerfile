FROM ubuntu:22.04

# Install Miniconda3
RUN apt-get update && apt-get install -y wget && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh

# Add Miniconda3 to PATH
ENV PATH /root/miniconda3/bin:$PATH

# Change the working directory
WORKDIR /eMOSIAC

# Copy the environment.yml file from your local machine into the Docker image
COPY environment.yml /eMOSIAC/environment.yml

# Create a Conda environment using the environment.yml
RUN conda env create -f environment.yml && \
    echo "source activate eMOSIAC" > ~/.bashrc && \
    /bin/bash -c "source activate eMOSIAC && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html && \
    pip install torch-geometric && \
    conda install -c conda-forge rdkit"


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "eMOSIAC", "/bin/bash"]
