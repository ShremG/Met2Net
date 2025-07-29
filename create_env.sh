conda create -n openstl python=3.10.8
conda activate openstl
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install lightning
conda install xarray==0.19.0
pip install hickle
pip install decord
pip install fvcore
pip install lpips
pip install scikit-image==0.19.3
pip install tqdm
pip install timm
pip install einops
pip install opencv-python
pip install opencv-python-headless
pip install matplotlib
pip install torchinfo
pip install netcdf4
pip install dask