from models import vggnet, resnet, wrn, MLP
import torchvision.models as models

""" 
    生成模型列表
"""


class ModelGeneration(object):

    def __init__(self, args, size, num_class):
        self.args = args
        self.size = size
        self.num_class = num_class

    def getting_models(self):
        models = []

        if self.args.model == 'VGG':
            model = vggnet.VGG(16, self.num_class)
        elif self.args.model == 'res':
            if self.args.dataset == 'cifar10':
                # model = large_resnet.ResNet18()
                model = resnet.ResNet(50, self.num_class)
            elif self.args.dataset == 'imagenet':
                model = models.resnet18()
        elif self.args.model == 'wrn':
            model = wrn.Wide_ResNet(28, 10, 0, self.num_class)
        elif self.args.model == 'mlp':
            if self.args.dataset == 'emnist':
                model = MLP.MNIST_MLP(47)

        # 分发给节点的模型列表
        for i in range(self.size):
            models.append(model)

        return models
