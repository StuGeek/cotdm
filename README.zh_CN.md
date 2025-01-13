# CoTDM

[[English]](README.md) | [[中文]](./README.zh_CN.md)

CoTDM是一个大语言模型推理系统。它通过一开始只在GPU显存中预部署模型的一部分参数层来降低部署成本，并使用了分时复用和并行加载技术有效地减少了流水线停顿，降低了推理延迟。

## 环境

+ Ubuntu 22.04
+ CUDA 12.4
+ Python 3.11
+ conda 23.10.0

## 安装

1. 在项目文件夹的根目录`CoTDM`中，输入命令：

```bash
pip install -r requirements.txt
```

2. 更改`global_data/`文件夹中`global_config.py`文件中`DATA_SAVE_DIR`的路径，这个路径存放项目相关的数据文件，例如：

```python
DATA_SAVE_DIR = "./"
```

3. 获取数据集，在`DATA_SAVE_DIR`目录下，输入

```bash
wget https://azurepublicdatasettraces.blob.core.windows.net/azurepublicdatasetv2/azurefunctions_dataset2019/azurefunctions-dataset2019.tar.xz
mkdir azurefunctions-dataset2019
tar -Jxvf azurefunctions-dataset2019.tar.xz -C azurefunctions-dataset2019
```

4. 获取实验所需的模型，在`CoTDM/submit/`文件夹下，输入命令：

```bash
sh submit_model.sh
```

## 运行实验

1. 在`CoTDM/experiments/`文件夹下，输入命令：

```bash
sh exp1.sh
sh exp2.sh
sh exp3.sh
```

2. 画图，在`CoTDM/experiments/`文件夹下，输入命令：

```bash
python plot_exp1_nor.py
python plot_exp2.py
python plot_exp3.py
```

实验结果图片可以在文件夹`CoTDM/experiments/`中找到

## 用法

在`CoTDM/`文件夹下输入命令启动服务端：

```bash
python server.py
```

客户端提交模型：

```bash
python client/client_submit_model.py --name bert-large-uncased
```

客户端发送请求：

```bash
python client/client_single_inf_req.py --name bert-large-uncased
```

或者

```bash
python client/client_workload.py
```