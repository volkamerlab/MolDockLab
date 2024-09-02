g#!/bin/bash

USE_GIT=0

while getopts "g" opt; do
  case $opt in
    g)
      USE_GIT=1
      ;;
    \?)
      echo "Usage: $0 [-g]" >&2
      exit 1
      ;;
  esac
done
BASEDIR=$PWD


# Check for the existence of required utilities

function check_dependency() {
    local command_name="$1"
    if ! command -v "$command_name" &> /dev/null; then
        echo "ERROR: $command_name is not installed, please install manually. Please run : "sudo apt-get install $command_name""
        exit 1
    fi
}

check_dependency "wget"
if [ $USE_GIT -eq 1 ]; then
    check_dependency "git"
fi
check_dependency "unzip"
check_dependency "gcc"


###############################################################

echo -e """
###############################################################
# Installing MolDockLab
###############################################################
"""

# install dependencies for xgboost, GWOVina & MGLTools
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "msys"* ]]; then
    # dependencies for mac and windows
    echo -e "MolDockLab is not compatible with Mac OS or Windows!"
    exit

elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo -e "Detected Linux OS!"

fi

###############################################################

echo -e """
###############################################################
# Verifying conda install or installing miniconda3 if not found
###############################################################
"""

# check if conda is installed, and install miniconda3 if not

# if conda is not a recognised command then download and install
if ! command -v conda &> /dev/null; then
    
    echo -e "No conda found - installing..."
    mkdir -p $HOME/miniconda3
    cd $HOME/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh --no-check-certificate -q --show-progress
    # install miniconda3
    cd $HOME/miniconda3 && chmod -x miniconda.sh
    cd $BASEDIR && bash $HOME/miniconda3/miniconda.sh -b -u -p $HOME/miniconda3

    # remove the installer
    rm -f $HOME/miniconda3/miniconda.sh

    # define conda installation paths
    CONDA_PATH="$HOME/miniconda3/bin/conda"
    CONDA_BASE=$BASEDIR/$HOME/miniconda3
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo -e "Found existing conda install!"
    # if conda not installed then find location of existing installation
    CONDA_PATH=$(which conda)
    CONDA_BASE=$(conda info --base)
    CONDA_SH=$CONDA_BASE/etc/profile.d/conda.sh
fi

###############################################################

echo -e """
###############################################################
# installing the MolDockLab conda environment
###############################################################
"""

# source the bash files to enable conda command in the same session
if test -f ~/.bashrc; then
    source ~/.bashrc
fi

if test -f ~/.bash_profile; then
    source ~/.bash_profile
fi

# initiate conda
$CONDA_PATH init bash

# source the conda shell script once initiated
source $CONDA_SH

# configure conda to install environment quickly and silently
$CONDA_PATH config --set auto_activate_base false
$CONDA_PATH config --set ssl_verify False

# create the conda environment
ENV_NAME="moldocklab"

if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping creation."
    conda activate moldocklab
else
    conda create -n $ENV_NAME python=3.10 -y
    conda deactivate
    conda activate $ENV_NAME

    conda config --add channels conda-forge

    conda install rdkit ipykernel scipy spyrmsd kneed scikit-learn-extra molvs seaborn xgboost openbabel docopt chembl_structure_pipeline tqdm plip ambertools schrodinger::pymol bioconda/label/cf201901::muscle -q -y

    echo -e """
    ###############################################################
    # Installing Pip packages, please wait...
    ###############################################################
    """

    pip install pymesh oddt biopandas redo MDAnalysis==2.0.0 prody==2.1.0 dgl tensorflow meeko posebusters hdbscan e3nn "fair-esm[esmfold]" pydantic dgl==1.1.3 plotly bravado black pybel -q

    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q

    pip install torch_scatter torch_sparse torch_spline_conv torch_cluster torch_geometric -q
    
    pip install -U kaleido
    echo -e """
    ###############################################################
    # Finished installing pip packages
    ###############################################################
    """

fi

###############################################################
MolDockLab_FOLDER=$(pwd)
cd $MolDockLab_FOLDER
###############################################################

echo -e """
###############################################################
# Downloading Executables...
###############################################################
"""
if [[ ! -d $MolDockLab_FOLDER/software ]]; then
    mkdir -p $MolDockLab_FOLDER/software
    cd $MolDockLab_FOLDER/software
else
    cd $MolDockLab_FOLDER/software
fi

if [[ ! -f $MolDockLab_FOLDER/software/gnina ]]; then
    echo -e "\nDownloading GNINA!"
    wget https://github.com/gnina/gnina/releases/latest/download/gnina --no-check-certificate  -q --show-progress
    chmod +x gnina
fi

if [[ ! -f $MolDockLab_FOLDER/software/PLANTS ]]; then
    echo -e "\nPLANTS not found in software folder, if you want to use it, please see documentation for a link to register and download it!"
fi

if [[ ! -d $MolDockLab_FOLDER/software/DiffDock ]]; then
    echo -e "\nCloning DiffDock!"
    cd $MolDockLab_FOLDER/software
    git clone git@github.com:mbackenkoehler/DiffDock.git
    cd $MolDockLab_FOLDER  # Return to the original directory
fi


if [[ ! -f $MolDockLab_FOLDER/software/smina.static ]]; then
    echo -e "\nDownloading Lin_F9!"
    cd $MolDockLab_FOLDER/software
    wget https://github.com/cyangNYU/Lin_F9_test/raw/master/smina.static --no-check-certificate -q --show-progress
    cd $MolDockLab_FOLDER
    chmod +x smina.static
fi


if [[ ! -d $MolDockLab_FOLDER/software/gypsum_dl-1.2.1 ]]; then
    echo -e "\nDownloading GypsumDL!"
    cd $MolDockLab_FOLDER/software
    wget https://github.com/durrantlab/gypsum_dl/archive/refs/tags/v1.2.1.tar.gz --no-check-certificate -q --show-progress
    tar -xf v1.2.1.tar.gz
    rm v1.2.1.tar.gz
    cd $MolDockLab_FOLDER
fi

if [[ ! -d $MolDockLab_FOLDER/software/SCORCH ]]; then
    echo -e "\nCloning SCORCH!"
    cd $MolDockLab_FOLDER/software
    git clone git@github.com:hamzaibrahim21/SCORCH.git
    cd $MolDockLab_FOLDER  # Return to the original directory
fi

if [[ ! -d $MolDockLab_FOLDER/software/PLIPify ]]; then
    echo -e "\nCloning PLIPify!"
    cd $MolDockLab_FOLDER/software
    git clone git@github.com:hamzaibrahim21/plipify.git
    cd $MolDockLab_FOLDER  # Return to the original directory
fi

if [[ ! -d $MolDockLab_FOLDER/software/RTMScore ]]; then
    echo -e "\nCloning RTMScore!"
    cd $MolDockLab_FOLDER/software
    git clone git@github.com:sc8668/RTMScore.git
    cd $MolDockLab_FOLDER  # Return to the original directory
fi

cd $BASEDIR


echo -e """
###############################################################
# MolDockLab installation complete
###############################################################
"""
###############################################################
echo -e """
###############################################################
# Checking installation success
###############################################################
"""

# Check if conda environment is present in the list of environments
if conda env list | grep -q $ENV_NAME; then
    echo -e "\nMolDockLab conda environment is present!"
else
    echo -e "\nINSTALLATION ERROR : MolDockLab conda environment is not present!"
fi

# Check if required packages are installed in the $ENV_NAME environment
required_packages=("rdkit" "ipykernel" "scipy" "spyrmsd" "kneed" "scikit-learn-extra" "molvs" "seaborn" "xgboost" "openbabel" "pymesh" "oddt" "biopandas" "redo" "MDAnalysis==2.0.0" "prody==2.1.0" "dgl" "tensorflow" "meeko" "posebusters" "torch" "torchvision" "torchaudio" "torch_scatter" "torch_sparse" "torch_spline_conv" "torch_cluster" "torch_geometric" "hdbscan" "e3nn")

for package in "${required_packages[@]}"; do
    if conda list -n $ENV_NAME "$package" &> /dev/null; then
        echo -e "$package is installed in the $ENV_NAME environment!"
    else
        echo -e "\nINSTALLATION ERROR : $package is not installed in the $ENV_NAME environment!"
    fi

conda activate $ENV_NAME
cd $MolDockLab_FOLDER

done
