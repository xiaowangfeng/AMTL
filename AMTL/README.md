# AMTL-PotNet
Official code repository of paper “[Adaptive Multi-Task Learning for Low-Resource Material
Property Prediction” by Zexi Yang, Qi Yu, Boran Li, Changfu Zuo,  and JIying Liu. 



## Environment Setup

- We'll use `conda` to install dependencies and set up the environment. We recommend using the [Python 3.9 Miniconda installer](https://docs.conda.io/en/latest/miniconda.html#linux-installers).
- After installing `conda`, run

```shell
conda env create -f environment.yml
```

- Then run below code to activate the environment

```
conda activate potnet
```

- We are using `torch-geometric` for the implementation of our graph neural networks. Installation of `torch-geometric` via conda seems problematic so we install it explicitly using pip as

```shell
pip install torch-scatter==2.1.0 torch-sparse==0.6.16 torch-cluster==1.6.0 torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.12.1%2Bcu116.html
```

and

```shell
pip install torch-geometric
```

- Be aware that we are using an old version of JARVIS toolkits `jarvis-tools==2022.9.16`. The newest JARVIS toolkits will contain new versions of datasets that include more data than the one we present in the paper.

## Running Summation Algorithm

- To run the summation algorithm, please run below commands in order to install the algorithm package (remember to replace the `TARGET_PATH` with your own destination.)

```shell
cd functions
tar xzvf gsl-latest.tar.gz
cd gsl-2.7.1
./configure --prefix=TARGET_PATH
make
make install
```

- Then edit `~/.bashrc` by adding

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TARGET_PATH/lib/
```

and 

```shell
source ~/.bashrc
```

or create another terminal.

- Now we back to `functions` directory and run

```shell
python setup.py build_ext --inplace
```

- Then the algorithm is installed as Cython package. A simple way to test if it is successfully installed is to run below in the root directory.

```shell
python test_algorithm.py
```

## Using Summation Algorithm

In this code base, we provide six infinite summations of

- Coulomb potential
- London dispersion potential
- Pauli repulsion potential
- Lennard-Jones potential
- Morse potential
- Screened Coulomb potential

and they are achieved in `algorithm.py` by

- `epstein` referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0}\frac{e^{2\pi i \mathbf{w} \cdot \mathbf{L} \mathbf{k}}}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^{2p}}$
- `zeta` referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0}\frac{1}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^{2p}}$
- `exp` referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d } e^{-\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert}$
- `lj` referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0}(\frac{\sigma^{12}}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^{12}} - \frac{\sigma^6}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert^6} )$
- `morse` referring to summation $\sum_{\mathbf{k}\in \mathbb{Z}^d} (e^{-2\alpha (\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert - r_e) } - 2e^{-\alpha (\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert - r_e)})$
- `screened_coulomb` referring to summation $\sum_{\mathbf{k}\in\mathbb{Z}^d, \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert\ne 0} \frac{e^{-\alpha \Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert}}{\Vert \mathbf{L}\mathbf{k}+\mathbf{v} \Vert} $

Each function requires the input of vectors `v` and a lattice matrix `Omega`, alongside a particular `param` and dimension `d`. We've also incorporated parameter `R`, which denotes half of the grid's length, and `verbose` which, when enabled, conducts corresponding error bound calculations.

For executing a single computation in parallel, set the `parallel` parameter to `True`. By default, the program will use `NUM_CPUS` (set to 32) for concurrent computation. However, keep in mind that during data processing within our model, this setting is overridden to `False`. This is due to the fact that we've applied a more efficient parallelization method via `pandarallel`.

The summation is computed over the grid instead of the ellipsoid, as we found this is much more efficient for computing summations of multiple vectors by numpy vectorization. Also, currently, we have implemented the error bound computation for `epstein`, `zeta`, and `exp` which are used in our model. For details of error bound computation, please refer to Appendix C.5 in our paper.



## Train on Custom Dataset

We are supporting custom datasets in the same format as datasets in [JARVIS Leaderboard](https://github.com/usnistgov/jarvis_leaderboard/tree/main) [4]. Once the corresponding dataset is prepared, use the script

```shell
python main.py --config configs/potnet.yaml --output_dir xxx --checkpoint xxx --data_root xxx
```

and change the target name as `target` in `potnet.yaml` such like

```yaml
dataset: dft_3d
target: target
```

And here `data_root` denotes the path of the custom dataset, where dataset information `dataset_info.json` and targets `id_prop.csv` are included. The id_prop.csv must contain three columns: the first column is the structure name, and the second and third columns are the target values. Then our code will read the dataset information from `dataset_info.json` and `id_prop.csv` in the dataset directory and then read the data from `data_root`. Note that a dataset in the JARVIS Leaderboard can be generated by [`jarvis_populate_data.py`](https://github.com/usnistgov/jarvis_leaderboard/blob/main/jarvis_leaderboard/jarvis_populate_data.py). To generate the same dataset format based on [`jarvis_populate_data.py`](https://github.com/usnistgov/jarvis_leaderboard/blob/main/jarvis_leaderboard/jarvis_populate_data.py) to accommodate our code, it is recommended to

- Generate the crystal structures that can be read by JARVIS toolkits and their corresponding properties to predict
- Predefine the train-val-test split of your dataset and write the crystal ids and targets in order in `id_prop.csv`



## Acknowledgement

The underlying training part is based on [PotNet]([AIRS/OpenMat/PotNet at main · divelab/AIRS](https://github.com/divelab/AIRS/tree/main/OpenMat/PotNet)) and  [CGCNN][txie-93/cgcnn: Crystal graph convolutional neural networks for predicting material properties.]. This work was supported in part by the  National Natural Science Foundation of China under Grant 62101572 and in part by Research Program of National University of Defense Technology ZK21-16.



## 
