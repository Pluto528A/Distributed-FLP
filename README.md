# Distributed-FLP

## Project Overview
Distributed-FLP is a distributed federated learning platform designed to achieve efficient distributed machine learning training. The project encompasses a complete process from parameter initialization, network structure construction, data management, model generation to simulated training, supporting various models such as ViT, ResNet, VGGNet, etc.


## Environment Requirements
- Python 3.11.8
- PyTorch 2.0.0
- torchvision 0.15.1
- vit-pytorch 1.10.1

## Install dependencies
```bash
pip install -r requirements.txt  # Make sure the requirements.txt file contains all dependencies
```

## Usage
### Run the main program
```bash
python main.py --aggRound 1 --size 31 --epoch 101 --numClass 10 --dataset cifar10
```
