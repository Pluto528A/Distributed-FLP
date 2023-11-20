import torch.optim as optim
import torch.nn as nn

class Simulation(object):
    """ Dataset-like object, but only access a subset of it. """

    def __init__(self, args, graphNetwork, train_loaders, test_loaders, models):
        self.args = args
        self.graphNetwork = graphNetwork
        self.train_loaders = train_loaders
        self.test_loaders = test_loaders
        self.models = models

        self.optimizers = []
        self.criterions = []

        for i in self.graphNetwork.size:
            self.optimizers[i] = optim.SGD(models[i].parameters(),
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=5e-4,
                                            nesterov=args.nesterov,
                                            dampening=0)
            self.criterion[i] = nn.CrossEntropyLoss()

    def run(self):
        print("开始训练")
