# Create and activate Conda environment
conda create --name R r-base=4.4.2
conda activate R

# Install required R packages via Conda
conda install -c conda-forge r-devtools r-glmnet r-ggplot2

# Start R to install bestsubset from GitHub
R -e "library(devtools); install_github(repo='ryantibs/best-subset', subdir='bestsubset')"

# if this fails try instead
# R
# > options(download.file.method = "libcurl")
# > options(timeout = 300)  # Increase timeout
# > install_github("ryantibs/best-subset", subdir="bestsubset", quiet=FALSE)
