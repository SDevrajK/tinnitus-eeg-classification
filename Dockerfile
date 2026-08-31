# Use miniconda3 as base image
FROM continuumio/miniconda3

# Set working directory
WORKDIR /workspace

# Copy environment file
COPY environment.yml .

# Create conda environment
RUN conda env create -f environment.yml

# Install additional dependencies not in environment.yml
RUN conda run -n tinnitus-eeg pip install scikit-learn joblib torch --index-url https://download.pytorch.org/whl/cpu
RUN conda run -n tinnitus-eeg pip install braindecode captum shap

# Copy the entire repository into the image
COPY . .

# Set default environment
ENV CONDA_DEFAULT_ENV=tinnitus-eeg

# Set default shell to run commands in the conda environment
SHELL ["conda", "run", "-n", "tinnitus-eeg", "/bin/bash", "-c"]

# Default command - start a shell
CMD ["/bin/bash"]