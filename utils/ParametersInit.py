"""
    参数设置
"""
import argparse

def para_init():
    # 设置参数
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    """
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        choices=[
            "mnist",
            "cifar10",
            "cifar100",
            "synthetic",
            "femnist",
            "emnist",
            "fmnist",
            "celeba",
            "medmnistS",
            "medmnistA",
            "medmnistC",
            "covid19",
            "svhn",
            "usps",
            "tiny_imagenet",
            "cinic10",
            "domain",
        ],
        default="cifar10",
    )"""
    parser.add_argument('--name', default="DecenSGD_CIFAR10", type=str, help='experiment name')
    parser.add_argument('--description', default="DecenSGD_CIFAR10", type=str, help='experiment description')

    parser.add_argument('--model', default="res", type=str, help='model name: res/VGG/wrn')
    parser.add_argument('--lr', default=0.8, type=float, help='learning rate')
    parser.add_argument('--momentum', default=0.0, type=float, help='momentum')
    parser.add_argument('--epoch', '-e', default=300, type=int, help='total epoch')
    parser.add_argument('--bs', default=64, type=int, help='batch size on each worker')
    parser.add_argument('--warmup', default=True, action='store_true', help='use lr warmup or not')
    parser.add_argument('--nesterov', default=False, action='store_true', help='use nesterov momentum or not')

    parser.add_argument('--budget', default=0.5, type=float, help='comm budget')
    parser.add_argument('--graphid', default=0, type=int, help='the idx of base graph')

    parser.add_argument('--dataset', default='cifar10', type=str, help='the dataset')
    parser.add_argument('--datasetRoot', default='./dataset', type=str, help='the path of dataset')
    parser.add_argument('--p', '-p', default=True, action='store_true', help='partition the dataset or not')
    parser.add_argument('--savePath', default='./saveModel', type=str, help='save path')
    parser.add_argument('--save', default=True, action='store_true', help='save medal or not')

    parser.add_argument('--compress', default=False, action='store_true', help='use chocoSGD or not')
    parser.add_argument('--consensus_lr', default=0.1, type=float, help='consensus_lr')
    parser.add_argument('--randomSeed', default=1234, type=int, help='random seed')

    args = parser.parse_args()

    return args
