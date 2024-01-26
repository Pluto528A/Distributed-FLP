"""
    分布式机器学习环境中进行通信和模型参数同步
"""


class Communicator(object):

    def __init__(self, graphNetwork):
        self.graphNetwork = graphNetwork

    def communicate(self, model):
        # 将所有模型参数堆叠成一个张量列表
        self.tensor_list = list()
        for param in model.parameters():
            self.tensor_list.append(param.data)

        # 预处理
        self.prepare_comm_buffer()

        # 交流在这里发生、记录通信时间
        comm_time = self.averaging()

        # 更新本地模型
        self.reset_model()

        return comm_time

    def prepare_comm_buffer(self):
        raise NotImplemented

    def averaging(self):
        raise NotImplemented

    def reset_model(self):
        raise NotImplemented

