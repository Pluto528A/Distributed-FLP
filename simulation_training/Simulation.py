import torch.optim as optim
import torch.nn as nn
import time
import torch
from utils import TrainUtils, RecordUtils

""" 
    模拟训练
"""


class Simulation(object):

    def __init__(self, args, graphNetwork, train_loaders, test_loaders, models):
        self.args = args
        self.graphNetwork = graphNetwork
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.models = models

        self.optimizers = []
        self.criterions = []

        self.losses = []
        self.top1s = []

        self.recorder = []

        # 所有模型的参数列表
        self.tensor_lists = []

        # 初始化 优化器、损失函数
        for i in range(self.graphNetwork.size):
            optimizer = optim.SGD(self.models[i].parameters(),
                                  lr=self.args.lr,
                                  momentum=self.args.momentum,
                                  weight_decay=5e-4,
                                  nesterov=self.args.nesterov,
                                  dampening=0)
            self.optimizers.append(optimizer)

            criterion = nn.CrossEntropyLoss()
            self.criterions.append(criterion)

            self.losses.append(TrainUtils.AverageMeter())
            self.top1s.append(TrainUtils.AverageMeter())

            self.recorder.append(RecordUtils.Recorder(self.args, i))

            tensor_list = list()
            for param in self.models[i].parameters():
                tensor_list.append(param.data.clone())
            self.tensor_lists.append(tensor_list)


        # 要传给相邻节点的模型参数
        sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).cpu()
        # 接受相邻节点传来的模型参数
        self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]

    def run(self):
        print("开始训练")

        for epoch in range(self.args.epoch):
            print(f"训练轮数： {epoch:0>3} / {self.args.epoch}")
            # self.tensor_lists = [[] for _ in range(self.graphNetwork.size)]

            for i in range(self.graphNetwork.size):
                self.models[i].train()
                for batch_idx, (data, target) in enumerate(self.train_loaders[i]):
                    # 加载数据
                    # data, target = data.cuda(non_blocking = True), target.cuda(non_blocking = True)

                    # 前向传播
                    output = self.models[i](data)
                    loss = self.criterions[i](output, target)

                    # 记录训练损失和准确率
                    acc1 = TrainUtils.comp_accuracy(output, target)
                    self.losses[i].update(loss.item(), data.size(0))
                    self.top1s[i].update(acc1[0].item(), data.size(0))
                    # if batch_idx % 10 == 0:
                    #     print(
                    #         f"{i}  第 {batch_idx} 次的 acc 为 ：{self.top1s[i].avg} - {self.top1s[i].val} - {self.top1s[i].count}")

                    # 反向传播
                    loss.backward()
                    update_learning_rate(self.args, self.optimizers[i], epoch, itr=batch_idx,
                                         itr_per_epoch=len(self.train_loaders[i]))

                    # 梯度更新
                    self.optimizers[i].step()
                    self.optimizers[i].zero_grad()

            sendModel = TrainUtils.flatten_tensors(self.tensor_lists[0]).cpu()
            self.receiveModels = [torch.zeros_like(sendModel) for _ in range(self.graphNetwork.size)]

            # 将本地模型传给相邻节点
            for i in range(self.graphNetwork.size):
                # 要传给相邻节点的模型参数
                self.tensor_lists[i] = []
                for param in self.models[i].parameters():
                    self.tensor_lists[i].append(param.data.clone())

                # 将所有模型参数堆叠成一个张量列表
                sendModel = TrainUtils.flatten_tensors(self.tensor_lists[i]).cpu()

                # 加上自己的模型参数，后面加权平均
                # self.receiveModels[i] = sendModel

                for j in range(self.graphNetwork.size):
                    if self.graphNetwork.matrix[i][j] != -1:
                        # 接收邻居节点传来的模型参数
                        self.receiveModels[j].add_(sendModel)

            # 对传来的模型参数加权平均
            for i in range(self.graphNetwork.size):

                # 将所有模型参数堆叠成一个张量列表
                sendModel = TrainUtils.flatten_tensors(self.tensor_lists[i]).cpu()

                self.receiveModels[i].add_(sendModel)

                # 加上本身
                self.receiveModels[i].div_(self.graphNetwork.degree[i] + 1)

                # 更新模型
                with torch.no_grad():
                    for new_data, param in zip(TrainUtils.unflatten_tensors(  # 将一个或多个可迭代对象（例如列表、元组、字符串等）按照相同索引的元素组合成元组对
                            self.receiveModels[i],  # .cuda(),
                            self.tensor_lists[i]),
                            self.models[i].parameters()):
                        param.data.copy_(new_data)  # 赋值

                    # for param, new_data in zip(self.models[i].parameters(), self.tensor_lists[i]):
                    #     param.data.copy_(new_data)

                test_acc = TrainUtils.test(self.models[i], self.test_loaders[i])
                print(f"{epoch:0>3}  ID: %d, epoch: %.3f, loss: %.3f, train_acc: %.3f, test_acc: %.3f" % (i, epoch,
                                                                                                          self.losses[
                                                                                                              i].avg,
                                                                                                          self.top1s[
                                                                                                              i].avg,
                                                                                                          test_acc))

                self.recorder[i].add_new(self.top1s[i].avg, self.losses[i].avg, test_acc)

                self.losses[i].reset()
                self.top1s[i].reset()

            if epoch % 10 == 0:
                for i in range(self.graphNetwork.size):
                    self.recorder[i].save_to_file()


def update_learning_rate(args, optimizer, epoch, itr=None, itr_per_epoch=None, scale=1):
    """
    Update learning rate with linear warmup and exponential decay.

    Args:
        args (namespace): 参数对象，包含学习率等信息。
        optimizer (torch.optim.Optimizer): 优化器。
        epoch (int): 当前训练的 epoch 数。
        itr (int): 当前迭代数（可选）。
        itr_per_epoch (int): 每个 epoch 的迭代数（可选）。
        scale (int): 缩放因子（可选）。

    Notes:
        1) Linearly warmup to reference learning rate (5 epochs)
        2) Decay learning rate exponentially (epochs 30, 60, 80)
        ** note: args.lr is the reference learning rate from which to scale up
        ** note: minimum global batch-size is 256
    """
    base_lr = 0.1  # 基础学习率
    target_lr = args.lr  # 目标学习率
    lr_schedule = [100, 150]  # 学习率衰减的阶段

    lr = None
    if args.warmup and epoch < 5:  # 如果启用了预热，并且当前 epoch 小于 5
        if target_lr <= base_lr:
            lr = target_lr
        else:
            assert itr is not None and itr_per_epoch is not None
            count = epoch * itr_per_epoch + itr + 1
            incr = (target_lr - base_lr) * (count / (5 * itr_per_epoch))
            lr = base_lr + incr
    else:
        lr = target_lr
        for e in lr_schedule:
            if epoch >= e:
                lr *= 0.1  # 指数衰减学习率

    if lr is not None:
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # print('Updating learning rate to {}'.format(lr))
