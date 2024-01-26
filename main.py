
from utils import ParametersInit
from network_structure.GraphNetwork import GraphNetwork
from data_management.DataGeneration import DataGeneration
from models.ModelGeneration import ModelGeneration
from simulation_training.Simulation import Simulation

if __name__ == '__main__':
    # 参数设置
    args = ParametersInit.para_init()

    # 分布式网络图结构  节点个数：8
    graphNetwork = GraphNetwork(args, 8)

    # # 对图结构预处理
    # GP = GraphProcessor(graph)

    # 定义节点通讯器
    # communicator = Communicator(graphNetwork)

    # 加载数据列表 返回一个训练、测试的数据列表
    dataGeneration = DataGeneration(args, graphNetwork.size)
    train_loaders, test_loaders = dataGeneration.partition_dataset()

    # 加载模型列表
    modelGeneration = ModelGeneration(args, graphNetwork.size, 10)
    models = modelGeneration.getting_models()

    # 训练
    simulator = Simulation(args, graphNetwork, train_loaders, test_loaders, models)
    simulator.run()

