#!/bin/bash

# Define the base conda path and environment name
CONDA_BASE="/home/${USER}/anaconda3"
ENV_NAME="GeoSapiens"
PYTHON_VERSION="3.10"
PYTORCH_VERSION="pytorch-cuda=12.1"

# Update with the path to your local conda directory
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Function to check if conda environment exists
conda_env_exists() {
  conda env list | grep -q "$1"
}

# Function to print messages in green
print_green() {
  echo -e "\033[0;32m$1\033[0m"
}

# Remove the environment if it exists
if conda_env_exists "${ENV_NAME}"; then
  print_green "Environment '${ENV_NAME}' exists. Removing..."
  conda env remove -n "${ENV_NAME}"
fi

# Create the new environment and activate it
print_green "Creating environment '${ENV_NAME}'..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
conda activate "${ENV_NAME}"


# Install PyTorch, torchvision, torchaudio, and specific CUDA version
print_green "Installing PyTorch, torchvision, torchaudio, and CUDA..."
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 "${PYTORCH_VERSION}" -c pytorch -c nvidia -y

# Install additional Python packages
print_green "Installing additional Python packages..."

pip install chumpy scipy munkres tqdm cython numpy==1.26.4 pandas fsspec yapf==0.40.1 matplotlib packaging omegaconf ipdb ftfy regex albumentations==1.4.24 tensorboard gdown

# Change directory to the root of the repository
cd "$(dirname "$0")/.."

# Function to install a package via pip with editable mode and verbose output
pip_install_editable() {
  print_green "Installing $1..."
  cd "$1" || exit
  pip install -e . -v
  cd - || exit
  print_green "Finished installing $1."
}

# Install engine
pip_install_editable "engine"

# Install mmcv, we need to uninstall the mmengine such that we could use the editable version
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

pip uninstall mmengine
# Install pretrain
pip_install_editable "pretrain"

# Install pose
pip_install_editable "pose"

# Install det
pip_install_editable "det"

print_green "Installation done!"
