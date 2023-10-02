# Exploring Physical Latent Spaces for High-Resolution Flow Restoration

This file will guide the readers through the steps to reproduce the results presented in our submission.

We provide the code and data for the *Karman vortex street* and *forced turbulence* scenarios.

**Warning**: Some configurations (eg: RTX 3000 series) might not work correctly with the setup we are providing.

## Table of Contents

- [Exploring Physical Latent Spaces for High-Resolution Flow Restoration](#exploring-physical-latent-spaces-for-high-resolution-flow-restoration)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Evaluation](#evaluation)
  - [Results](#results)


## Requirements

To install requirements:

```setup
* install Miniconda:
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  sh Miniconda3-latest-Linux-x86_64.sh

* activate the base conda env:
  eval "$(~/miniconda3/bin/conda shell.zsh hook)"

* (optional) update conda:
  conda update -n base -c defaults conda

* create a conda env:
  conda create -n pf2 tensorflow-gpu=2.4.1 numpy=1.19 cudnn=7.6.5

* activate the env:
  conda activate pf2

* install cuda toolkit:
  conda install -c conda-forge cudatoolkit-dev=10.1.243
  conda install gcc_linux-64=7.5.0 
  conda install gxx_linux-64=7.5.0 

* make symbolic links from bin to nvcc, g++, etc
  cd ~/miniconda3/envs/pf2/bin
  (optional) ln -s ~/miniconda3/envs/pf2/pkgs/cuda-toolkit/bin/nvcc nvcc
  ln -s ./x86_64-conda_cos6-linux-gnu-g++ g++
  ln -s ./x86_64-conda_cos6-linux-gnu-gcc gcc
```

  You might need to reactivate your environment after creating these symbolic links.

```setup
* conda activate pf2

* go to the folder containing the code of the submission:
  cd code
  git clone https://github.com/tum-pbs/PhiFlow.git
  cd PhiFlow
  git checkout 2.0.1

* in your PhiFlow folder, edit *setup.py* at l.80-81:

  self.nvcc = 'nvcc'
  self.cuda_lib = '/home/<username>/miniconda3/envs/pf2/lib/'

* compile this file:
  python3 setup.py tf_cuda
```

 **Only if the compilation is complete, do:**

```setup
* in phi/tf/_tf_backend.py, in the *grid_sample()* function (l. 171) add this condition at the very beginning:

  if extrapolation != 'boundary':
    return NotImplemented

* pip install gast==0.3.3

* make symbolic links to "phi" in the scripts folders:

  if not already in the "code" folder: cd code

  ln -s PhiFlow/phi .

  cd test_scripts
  ln -s ../PhiFlow/phi .

  cd ../metrics_scripts
  ln -s ../PhiFlow/phi .

  cd ../images_scripts
  ln -s ../PhiFlow/phi .

* go back to the "code" folder:
  cd ..

* install required packages
  conda install matplotlib
  conda install psutil
  conda install -c conda-forge pyfftw

```

## Evaluation

To generate the inferences of each model, run the following python scripts from the *code* folder:

```eval
python3 karman_generate_results.py
python3 forced_turb_generate_results.py
```

The inferences will be created in the *inferences* folder.

## Results

To generate the plots that summarize the metrics comparing the models performances, the "--mae", "--mse" and "--reduced" arguments enable to choose which metrics to plot.

The "reduced" argument will plot the mean absolute distance between the reduced states produced by each model and the linear down-sampling of the corresponding reference.

**All plots will be saved into the *metrics* folder.**

```plot
cd metrics_scripts
python3 karman_plot_metrics.py --mae --mse --reduced
python3 forced_turb_plot_metrics.py --mae --mse --reduced
```
