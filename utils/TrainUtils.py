import torch


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
        将梯度从不同部分的神经网络参数展平并连接，以便进行梯度更新操作（如优化算法中的梯度下降）或者用于某些分布式计算操作
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
        这个函数的作用是从一个1D缓冲区中还原多个张量，并确保它们的大小正确匹配。这在分布式深度学习中，特别是在梯度聚合等操作中非常有用。
        通常，它与 flatten_tensors 函数一起使用，前者用于将多个张量合并成一个1D缓冲区，后者用于将1D缓冲区还原为多个张量。
        这有助于在分布式环境中传输和同步模型参数或梯度。
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def comp_accuracy(output, target, topk=(1,)):
    """计算指定 k 值的 k 个最高预测值的准确度"""
    # 禁用梯度计算
    with torch.no_grad():
        # 获取topk中的最大值
        maxk = max(topk)
        # 获取目标标签的批次大小
        batch_size = target.size(0)

        # 从模型输出中获取topk的索引
        _, pred = output.topk(maxk, 1, True, True)
        # 将索引矩阵转置
        pred = pred.t()
        # 创建一个布尔矩阵，指示模型的预测是否正确
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # 初始化准确率结果列表
        res = []
        for k in topk:
            # 计算前k个准确的预测的数量
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # 将准确预测的数量转化为百分比
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    # correct = 0
    # total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        # inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        outputs = model(inputs)
        acc1 = comp_accuracy(outputs, targets)
        top1.update(acc1[0], inputs.size(0))
        # if batch_idx % 10 == 0:
        #     print(
        #         f"第 {batch_idx} 次的 acc 为 ：{top1.avg} - {top1.val} - {top1.count} - {top1.sum}")
    return top1.avg


class AverageMeter(object):
    def __init__(self):
        # 初始化函数，创建一个 AverageMeter 实例时会调用
        self.reset()
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 计数器

    def reset(self):
        # 重置函数，用于将计数器的各个属性重置为初始状态
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        # 更新函数，用于更新计数器的各个属性
        self.val = val  # 设置当前值为传入的值
        self.sum += val * n  # 将值乘以权重 n 并加到总和上
        self.count += n  # 更新计数器
        self.avg = self.sum / self.count  # 计算新的平均值
