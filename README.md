### Read me from mushfiq

##### Run the code
My GPU (NVIDIA 3050 Ti) did not support the CUDNN version required for Pytorch 1.3
I instaled pytorch 2.0 and python 3.8

To train the model with cough data and Bayesian CNN please run
``python main_bayesian.py``
After training to test the performance on test data please run
``python main_bayesian_test.py``

To train the CNN counterpart please run
``python main_frequentist_cough.py``
And to test its performance
``python main_frequentist_test.py``

##### Data downloading and processing
You need to download the EdgeAICough folder inside ./data
Folder structure should look like ./data/EdgeAICough/train.pt

To have these data you first need to download the `public_dataset.zip` from [here](https://drive.google.com/file/d/1vCR_-QN_1R65z__yl8GjKn73p5CoRkdO/view?usp=drive_link). This is different from original `public_dataset.zip` of the dataset website, as I divided them into train and test.
Extract the .zip file inside the folder: ./public_dataset/train/...

Then run the file ``process_cough_data.py``. It will generate four files inside ./data/EdgeAICough/
Now you are good to go with training and testing

##### My conda environment
My conda package list is given below:
```
(DF) mush@mush-XPS-15:~$ conda list
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                  2_kmp_llvm    conda-forge
albumentations            1.3.0                    pypi_0    pypi
anyio                     3.5.0            py38h06a4308_0  
appdirs                   1.4.4              pyhd3eb1b0_0  
argon2-cffi               21.3.0             pyhd3eb1b0_0  
argon2-cffi-bindings      21.2.0           py38h7f8727e_0  
asttokens                 2.2.1                    pypi_0    pypi
attrs                     22.1.0           py38h06a4308_0  
audioread                 3.0.1            py38h578d9bd_1    conda-forge
babel                     2.11.0           py38h06a4308_0  
backcall                  0.2.0              pyhd3eb1b0_0  
beautifulsoup4            4.12.0           py38h06a4308_0  
blas                      1.0                         mkl  
bleach                    4.1.0              pyhd3eb1b0_0  
bottleneck                1.3.5            py38h7deecbd_0    anaconda
brotli                    1.0.9                h5eee18b_7  
brotli-bin                1.0.9                h5eee18b_7  
brotlipy                  0.7.0           py38h27cfd23_1003  
bzip2                     1.0.8                h7b6447c_0  
ca-certificates           2023.12.12           h06a4308_0  
certifi                   2024.2.2         py38h06a4308_0  
cffi                      1.15.1           py38h5eee18b_3  
charset-normalizer        2.0.4              pyhd3eb1b0_0  
comm                      0.1.2            py38h06a4308_0  
contourpy                 1.0.5            py38hdb19cb5_0  
cryptography              39.0.1           py38h9ce1e76_0  
cuda-cudart               11.7.99                       0    nvidia
cuda-cupti                11.7.101                      0    nvidia
cuda-libraries            11.7.1                        0    nvidia
cuda-nvrtc                11.7.99                       0    nvidia
cuda-nvtx                 11.7.91                       0    nvidia
cuda-runtime              11.7.1                        0    nvidia
cycler                    0.11.0             pyhd3eb1b0_0  
cython                    0.29.33          py38h6a678d5_0  
dbus                      1.13.18              hb2f20db_0  
debugpy                   1.5.1            py38h295c915_0  
decorator                 5.1.1              pyhd3eb1b0_0  
defusedxml                0.7.1              pyhd3eb1b0_0  
efficientnet-pytorch      0.7.1                    pypi_0    pypi
entrypoints               0.4              py38h06a4308_0  
executing                 1.2.0                    pypi_0    pypi
expat                     2.4.9                h6a678d5_0  
ffmpeg                    4.3                  hf484d3e_0    pytorch
filelock                  3.9.0            py38h06a4308_0  
flit-core                 3.8.0            py38h06a4308_0  
fontconfig                2.14.1               h52c9d5c_1  
fonttools                 4.25.0             pyhd3eb1b0_0  
freetype                  2.12.1               h4a9f257_0  
gettext                   0.21.1               h27087fc_0    conda-forge
giflib                    5.2.1                h5eee18b_3  
glib                      2.69.1               he621ea3_2  
gmp                       6.2.1                h295c915_3  
gmpy2                     2.1.2            py38heeb90bb_0  
gnutls                    3.6.15               he1e5248_0  
gst-plugins-base          1.14.1               h6a678d5_1  
gstreamer                 1.14.1               h5eee18b_1  
h5py                      3.7.0            py38h737f45e_0  
hdf5                      1.10.6               h3ffc7dd_1  
icu                       58.2                 he6710b0_3  
idna                      3.4              py38h06a4308_0  
imageio                   2.26.1                   pypi_0    pypi
imgaug                    0.4.0                    pypi_0    pypi
importlib-metadata        6.0.0            py38h06a4308_0  
importlib_metadata        6.0.0                hd3eb1b0_0  
importlib_resources       5.2.0              pyhd3eb1b0_1  
intel-openmp              2021.4.0          h06a4308_3561  
ipykernel                 6.19.2           py38hb070fc8_0  
ipython                   8.11.0                   pypi_0    pypi
ipython_genutils          0.2.0              pyhd3eb1b0_1  
ipywidgets                8.0.4            py38h06a4308_0  
jedi                      0.18.2                   pypi_0    pypi
jinja2                    3.1.2            py38h06a4308_0  
joblib                    1.2.0            py38h06a4308_0  
jpeg                      9e                   h5eee18b_1  
json5                     0.9.6              pyhd3eb1b0_0  
jsonschema                4.17.3           py38h06a4308_0  
jupyter                   1.0.0            py38h06a4308_8  
jupyter_client            8.1.0            py38h06a4308_0  
jupyter_console           6.6.3            py38h06a4308_0  
jupyter_core              5.3.0            py38h06a4308_0  
jupyter_server            1.23.4           py38h06a4308_0  
jupyterlab                3.5.3            py38h06a4308_0  
jupyterlab_pygments       0.1.2                      py_0  
jupyterlab_server         2.22.0           py38h06a4308_0  
jupyterlab_widgets        3.0.5            py38h06a4308_0  
kiwisolver                1.4.4            py38h6a678d5_0  
krb5                      1.19.4               h568e23c_0  
lame                      3.100                h7b6447c_0  
lazy-loader               0.2                      pypi_0    pypi
lazy_loader               0.3              py38h06a4308_0  
lcms2                     2.12                 h3be6417_0  
ld_impl_linux-64          2.38                 h1181459_1  
lerc                      3.0                  h295c915_0  
libbrotlicommon           1.0.9                h5eee18b_7  
libbrotlidec              1.0.9                h5eee18b_7  
libbrotlienc              1.0.9                h5eee18b_7  
libclang                  10.0.1          default_hb85057a_2  
libcublas                 11.10.3.66                    0    nvidia
libcufft                  10.7.2.124           h4fbf590_0    nvidia
libcufile                 1.6.0.25                      0    nvidia
libcurand                 10.3.2.56                     0    nvidia
libcusolver               11.4.0.1                      0    nvidia
libcusparse               11.7.4.91                     0    nvidia
libdeflate                1.17                 h5eee18b_0  
libedit                   3.1.20221030         h5eee18b_0  
libevent                  2.1.12               h8f2d780_0  
libffi                    3.4.2                h6a678d5_6  
libflac                   1.4.3                h59595ed_0    conda-forge
libgcc-ng                 12.2.0              h65d4601_19    conda-forge
libgfortran-ng            11.2.0               h00389a5_1  
libgfortran5              11.2.0               h1234567_1  
libiconv                  1.16                 h7f8727e_2  
libidn2                   2.3.2                h7f8727e_0  
libllvm10                 10.0.1               hbcb73fb_5  
libllvm14                 14.0.6               hdb19cb5_3  
libnpp                    11.7.4.75                     0    nvidia
libnvjpeg                 11.8.0.2                      0    nvidia
libogg                    1.3.5                h27cfd23_1  
libopus                   1.3.1                h7b6447c_0  
libpng                    1.6.39               h5eee18b_0  
libpq                     12.9                 h16c4e8d_3  
libprotobuf               3.15.8               h780b84a_1    conda-forge
librosa                   0.10.1             pyhd8ed1ab_0    conda-forge
libsndfile                1.2.2                hc60ed4a_1    conda-forge
libsodium                 1.0.18               h7b6447c_0  
libstdcxx-ng              13.2.0               h7e041cc_5    conda-forge
libtasn1                  4.16.0               h27cfd23_0  
libtiff                   4.5.0                h6a678d5_2  
libunistring              0.9.10               h27cfd23_0  
libuuid                   1.41.5               h5eee18b_0  
libvorbis                 1.3.7                h7b6447c_0  
libwebp                   1.2.4                h11a3e52_1  
libwebp-base              1.2.4                h5eee18b_1  
libxcb                    1.15                 h7f8727e_0  
libxkbcommon              1.0.1                hfa300c1_0  
libxml2                   2.9.14               h74e7548_0  
libxslt                   1.1.35               h4e12654_0  
llvm-openmp               14.0.6               h9e868ea_0  
llvmlite                  0.40.0           py38he621ea3_0  
lxml                      4.9.1            py38h1edc446_0  
lz4-c                     1.9.4                h6a678d5_0  
markupsafe                2.1.1            py38h7f8727e_0  
matplotlib                3.7.1            py38h06a4308_0  
matplotlib-base           3.7.1            py38h417a72b_0  
matplotlib-inline         0.1.6            py38h06a4308_0  
mistune                   0.8.4           py38h7b6447c_1000  
mkl                       2021.4.0           h06a4308_640  
mkl-service               2.4.0            py38h7f8727e_0  
mkl_fft                   1.3.1            py38hd3c417c_0  
mkl_random                1.2.2            py38h51133e4_0  
mpc                       1.1.0                h10f8cd9_1  
mpfr                      4.0.2                hb69a4c5_1  
mpg123                    1.32.4               h59595ed_0    conda-forge
mpmath                    1.2.1            py38h06a4308_0  
msgpack-python            1.0.3            py38hd09550d_0  
munkres                   1.1.4                      py_0  
nbclassic                 0.5.4            py38h06a4308_0  
nbclient                  0.5.13           py38h06a4308_0  
nbconvert                 6.5.4            py38h06a4308_0  
nbformat                  5.7.0            py38h06a4308_0  
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.5.6            py38h06a4308_0  
nettle                    3.7.3                hbbd107a_1  
networkx                  2.8.4            py38h06a4308_1  
notebook                  6.5.3            py38h06a4308_0  
notebook-shim             0.2.2            py38h06a4308_0  
nspr                      4.33                 h295c915_0  
nss                       3.74                 h0370c37_0  
numba                     0.57.1           py38hd559b08_0    conda-forge
numexpr                   2.8.4            py38he184ba9_0    anaconda
numpy                     1.23.5           py38h14f4228_0  
numpy-base                1.23.5           py38h31eccc5_0  
opencv-python             4.7.0.72                 pypi_0    pypi
opencv-python-headless    4.7.0.72                 pypi_0    pypi
openh264                  2.1.1                h4ff587b_0  
openssl                   1.1.1w               h7f8727e_0  
packaging                 23.0             py38h06a4308_0  
pandas                    1.5.3                    pypi_0    pypi
pandocfilters             1.5.0              pyhd3eb1b0_0  
parso                     0.8.3              pyhd3eb1b0_0  
pcre                      8.45                 h295c915_0  
pexpect                   4.8.0              pyhd3eb1b0_3  
pickleshare               0.7.5           pyhd3eb1b0_1003  
pillow                    9.4.0            py38h6a678d5_0  
pip                       23.0.1           py38h06a4308_0  
pkgutil-resolve-name      1.3.10           py38h06a4308_0  
platformdirs              2.5.2            py38h06a4308_0  
ply                       3.11                     py38_0  
pooch                     1.4.0              pyhd3eb1b0_0  
prometheus_client         0.14.1           py38h06a4308_0  
prompt-toolkit            3.0.38                   pypi_0    pypi
prompt_toolkit            3.0.36               hd3eb1b0_0  
protobuf                  3.15.8           py38h709712a_0    conda-forge
psutil                    5.9.0            py38h5eee18b_0  
ptyprocess                0.7.0              pyhd3eb1b0_2  
pure_eval                 0.2.2              pyhd3eb1b0_0  
pycocotools               2.0.6            py38h26c90d9_1    conda-forge
pycparser                 2.21               pyhd3eb1b0_0  
pygments                  2.14.0                   pypi_0    pypi
pyopenssl                 23.0.0           py38h06a4308_0  
pyparsing                 3.0.9            py38h06a4308_0  
pyqt                      5.15.7           py38h6a678d5_1  
pyqt5-sip                 12.11.0          py38h6a678d5_1  
pyrsistent                0.18.0           py38heee7806_0  
pysocks                   1.7.1            py38h06a4308_0  
pysoundfile               0.12.1             pyhd8ed1ab_0    conda-forge
python                    3.8.16               h7a1cb2a_3  
python-dateutil           2.8.2              pyhd3eb1b0_0  
python-fastjsonschema     2.16.2           py38h06a4308_0  
python_abi                3.8                      2_cp38    conda-forge
pytorch                   2.0.0           py3.8_cuda11.7_cudnn8.5.0_0    pytorch
pytorch-cuda              11.7                 h778d358_3    pytorch
pytorch-mutex             1.0                        cuda    pytorch
pytz                      2022.7.1                 pypi_0    pypi
pywavelets                1.4.1                    pypi_0    pypi
pyyaml                    6.0                      pypi_0    pypi
pyzmq                     23.2.0           py38h6a678d5_0  
qt-main                   5.15.2               h327a75a_7  
qt-webengine              5.15.9               hd2b0992_4  
qtconsole                 5.4.0            py38h06a4308_0  
qtpy                      2.2.0            py38h06a4308_0  
qtwebkit                  5.212                h4eab89a_4  
qudida                    0.0.4                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
requests                  2.28.1           py38h06a4308_1  
scikit-image              0.20.0                   pypi_0    pypi
scikit-learn              1.2.2                    pypi_0    pypi
scipy                     1.9.1                    pypi_0    pypi
seaborn                   0.12.2           py38h06a4308_0    anaconda
send2trash                1.8.0              pyhd3eb1b0_1  
setuptools                65.6.3           py38h06a4308_0  
shapely                   2.0.1                    pypi_0    pypi
sip                       6.6.2            py38h6a678d5_0  
six                       1.16.0             pyhd3eb1b0_1  
sniffio                   1.2.0            py38h06a4308_1  
soupsieve                 2.4              py38h06a4308_0  
soxr                      0.1.3                h0b41bf4_3    conda-forge
soxr-python               0.3.7            py38h7f0c24c_0    conda-forge
sqlite                    3.41.1               h5eee18b_0  
stack-data                0.6.2                    pypi_0    pypi
stack_data                0.2.0              pyhd3eb1b0_0  
sympy                     1.11.1           py38h06a4308_0  
tensorboardx              2.5.1              pyhd8ed1ab_0    conda-forge
terminado                 0.17.1           py38h06a4308_0  
threadpoolctl             3.1.0                    pypi_0    pypi
tifffile                  2023.3.15                pypi_0    pypi
tinycss2                  1.2.1            py38h06a4308_0  
tk                        8.6.12               h1ccaba5_0  
toml                      0.10.2             pyhd3eb1b0_0  
tomli                     2.0.1            py38h06a4308_0  
torchtriton               2.0.0                      py38    pytorch
torchvision               0.15.0               py38_cu117    pytorch
tornado                   6.2              py38h5eee18b_0  
tqdm                      4.65.0                   pypi_0    pypi
traitlets                 5.9.0                    pypi_0    pypi
typing-extensions         4.4.0            py38h06a4308_0  
typing_extensions         4.4.0            py38h06a4308_0  
urllib3                   1.26.14          py38h06a4308_0  
wcwidth                   0.2.6                    pypi_0    pypi
webencodings              0.5.1                    py38_1  
websocket-client          0.58.0           py38h06a4308_4  
wheel                     0.38.4           py38h06a4308_0  
widgetsnbextension        4.0.5            py38h06a4308_0  
xz                        5.2.10               h5eee18b_1  
zeromq                    4.3.4                h2531618_0  
zipp                      3.11.0           py38h06a4308_0  
zlib                      1.2.13               h5eee18b_0  
zstd                      1.5.2                ha4553b6_0  
```



[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![Pytorch 1.3](https://img.shields.io/badge/pytorch-1.3.1-blue.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/LICENSE)
[![arxiv](https://img.shields.io/badge/stat.ML-arXiv%3A2002.02797-B31B1B.svg)](https://arxiv.org/abs/1901.02731)

We introduce **Bayesian convolutional neural networks with variational inference**, a variant of convolutional neural networks (CNNs), in which the intractable posterior probability distributions over weights are inferred by **Bayes by Backprop**. We demonstrate how our proposed variational inference method achieves performances equivalent to frequentist inference in identical architectures on several datasets (MNIST, CIFAR10, CIFAR100) as described in the [paper](https://arxiv.org/abs/1901.02731).

---------------------------------------------------------------------------------------------------------


### Filter weight distributions in a Bayesian Vs Frequentist approach

![Distribution over weights in a CNN's filter.](experiments/figures/BayesCNNwithdist.png)

---------------------------------------------------------------------------------------------------------

### Fully Bayesian perspective of an entire CNN

![Distributions must be over weights in convolutional layers and weights in fully-connected layers.](experiments/figures/CNNwithdist_git.png)

---------------------------------------------------------------------------------------------------------



### Layer types

This repository contains two types of bayesian lauer implementation:  
* **BBB (Bayes by Backprop):**  
  Based on [this paper](https://arxiv.org/abs/1505.05424). This layer samples all the weights individually and then combines them with the inputs to compute a sample from the activations.

* **BBB_LRT (Bayes by Backprop w/ Local Reparametrization Trick):**  
  This layer combines Bayes by Backprop with local reparametrization trick from [this paper](https://arxiv.org/abs/1506.02557). This trick makes it possible to directly sample from the distribution over activations.
---------------------------------------------------------------------------------------------------------



### Make your custom Bayesian Network?
To make a custom Bayesian Network, inherit `layers.misc.ModuleWrapper` instead of `torch.nn.Module` and use `BBBLinear` and `BBBConv2d` from any of the given layers (`BBB` or `BBB_LRT`) instead of `torch.nn.Linear` and `torch.nn.Conv2d`. Moreover, no need to define `forward` method. It'll automatically be taken care of by `ModuleWrapper`. 

For example:  
```python
class Net(nn.Module):

  def __init__(self):
    super().__init__()
    self.conv = nn.Conv2d(3, 16, 5, strides=2)
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU()
    self.fc = nn.Linear(800, 10)

  def forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    x = self.relu(x)
    x = x.view(-1, 800)
    x = self.fc(x)
    return x
```
Above Network can be converted to Bayesian as follows:
```python
class Net(ModuleWrapper):

  def __init__(self):
    super().__init__()
    self.conv = BBBConv2d(3, 16, 5, strides=2)
    self.bn = nn.BatchNorm2d(16)
    self.relu = nn.ReLU()
    self.flatten = FlattenLayer(800)
    self.fc = BBBLinear(800, 10)
```

#### Notes:
1. Add `FlattenLayer` before first `BBBLinear` block.  
2. `forward` method of the model will return a tuple as `(logits, kl)`.
3. `priors` can be passed as an argument to the layers. Default value is:  
```python
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-3, 0.1),  # (mean, std) normal_
}
```

---------------------------------------------------------------------------------------------------------

### How to perform standard experiments?
Currently, following datasets and models are supported.  
* Datasets: MNIST, CIFAR10, CIFAR100  
* Models: AlexNet, LeNet, 3Conv3FC  

#### Bayesian

`python main_bayesian.py`
* set hyperparameters in `config_bayesian.py`


#### Frequentist

`python main_frequentist.py`
* set hyperparameters in `config_frequentist.py`

---------------------------------------------------------------------------------------------------------



### Directory Structure:
`layers/`:  Contains `ModuleWrapper`, `FlattenLayer`, `BBBLinear` and `BBBConv2d`.  
`models/BayesianModels/`: Contains standard Bayesian models (BBBLeNet, BBBAlexNet, BBB3Conv3FC).  
`models/NonBayesianModels/`: Contains standard Non-Bayesian models (LeNet, AlexNet).  
`checkpoints/`: Checkpoint directory: Models will be saved here.  
`tests/`: Basic unittest cases for layers and models.  
`main_bayesian.py`: Train and Evaluate Bayesian models.  
`config_bayesian.py`: Hyperparameters for `main_bayesian` file.  
`main_frequentist.py`: Train and Evaluate non-Bayesian (Frequentist) models.  
`config_frequentist.py`: Hyperparameters for `main_frequentist` file.  

---------------------------------------------------------------------------------------------------------



### Uncertainty Estimation:  
There are two types of uncertainties: **Aleatoric** and **Epistemic**.  
Aleatoric uncertainty is a measure for the variation of data and Epistemic uncertainty is caused by the model.  
Here, two methods are provided in `uncertainty_estimation.py`, those are `'softmax'` & `'normalized'` and are respectively based on equation 4 from [this paper](https://openreview.net/pdf?id=Sk_P2Q9sG) and equation 15 from [this paper](https://arxiv.org/pdf/1806.05978.pdf).  
Also, `uncertainty_estimation.py` can be used to compare uncertainties by a Bayesian Neural Network on `MNIST` and `notMNIST` dataset. You can provide arguments like:     
1. `net_type`: `lenet`, `alexnet` or `3conv3fc`. Default is `lenet`.   
2. `weights_path`: Weights for the given `net_type`. Default is `'checkpoints/MNIST/bayesian/model_lenet.pt'`.  
3. `not_mnist_dir`: Directory of `notMNIST` dataset. Default is `'data\'`. 
4. `num_batches`: Number of batches for which uncertainties need to be calculated.  

**Notes**:  
1. You need to download the [notMNIST](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) dataset from [here](http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz).  
2. Parameters `layer_type` and `activation_type` used in `uncertainty_etimation.py` needs to be set from `config_bayesian.py` in order to match with provided weights. 

---------------------------------------------------------------------------------------------------------



If you are using this work, please cite:

```
@article{shridhar2019comprehensive,
  title={A comprehensive guide to bayesian convolutional neural network with variational inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1901.02731},
  year={2019}
}
```

```
@article{shridhar2018uncertainty,
  title={Uncertainty estimations by softplus normalization in bayesian convolutional neural networks with variational inference},
  author={Shridhar, Kumar and Laumann, Felix and Liwicki, Marcus},
  journal={arXiv preprint arXiv:1806.05978},
  year={2018}
}
}
```

--------------------------------------------------------------------------------------------------------
