# CoTDM

[[English]](README.md) | [[中文]](./README.zh_CN.md)

CoTDM is a multi-LLM inference system. It reduces deployment costs by deploying only a subset of a model’s parameter layers to GPU memory, and effectively reduces pipeline stalls and inference latency by using time-division multiplexing and parallel loading techniques.

## Environment

+ Ubuntu 22.04
+ CUDA 12.4
+ Python 3.11
+ conda 23.10.0

## Installation

1. In the root folder of `CoTDM` project, enter the command:

```bash
pip install -r requirements.txt
```

2. Change the path of `DATA_SAVE_DIR` in the file `global_data/global_config.py`, which stores project related data files, for example:

```python
DATA_SAVE_DIR = "./"
```

3. Get dataset. In the directory `DATA_SAVE_DIR`, enter the command:

```bash
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
mkdir azurefunctions-dataset2019
tar -Jxvf azurefunctions-dataset2019.tar.xz -C azurefunctions-dataset2019
```

4. Get the models required for the experiment. In the directory `submit/`, enter the command:

```bash
sh submit_model.sh
```

## Run experiments

1. In the directory `experiments/`, enter the command:

```bash
sh exp1.sh
sh exp2.sh
sh exp3.sh
```

2. Draw figures. In the directory `experiments/`, enter the command:

```bash
python plot_exp1_nor.py
python plot_exp2.py
python plot_exp3.py
```

The experiment result figures can be found in the folder `experiments/`.

## Usage

In the root folder of `CoTDM` project, enter the command to start the server:

```bash
python server.py
```

Submit a model on the client side:

```bash
python client/client_submit_model.py --name bert-large-uncased
```

Sending a request on the client side:

```bash
python client/client_single_inf_req.py --name bert-large-uncased
```

or

```bash
python client/client_workload.py
```