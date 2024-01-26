import os
import numpy as np
import torch


class Recorder(object):
    def __init__(self, args, id):
        # 用于存储记录指标的列表
        self.record_accuracy = list()
        self.record_timing = list()
        self.record_comp_timing = list()
        self.record_comm_timing = list()
        self.record_losses = list()
        self.record_trainacc = list()
        self.total_record_timing = list()

        # 存储输入参数，节点ID
        self.args = args
        self.id = id

        # 创建一个文件夹以保存记录
        self.saveFolderName = args.savePath + '_' + args.name + '_' + args.model
        if os.path.isdir(self.saveFolderName) == False and self.args.save:
            os.mkdir(self.saveFolderName)

    def add_new(self, top1, losses, test_acc):
        # 将新记录添加到列表中
        self.record_trainacc.append(top1)
        self.record_losses.append(losses)
        self.record_accuracy.append(test_acc)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-acc.log', self.record_accuracy, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-losses.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID_' + str(self.id) + '-tacc.log', self.record_trainacc, delimiter=',')
        with open(self.saveFolderName + '/ExpDescription', 'w') as f:
            f.write(str(self.args) + '\n')
            f.write(self.args.description + '\n')

    # def add_new(self, record_time, comp_time, comm_time, epoch_time, top1, losses, test_acc):
    #     # 将新记录添加到列表中
    #     self.total_record_timing.append(record_time)
    #     self.record_timing.append(epoch_time)
    #     self.record_comp_timing.append(comp_time)
    #     self.record_comm_timing.append(comm_time)
    #     self.record_trainacc.append(top1)
    #     self.record_losses.append(losses)
    #     self.record_accuracy.append(test_acc)
    # def save_to_file(self):
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-recordtime.log', self.total_record_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-time.log', self.record_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-comptime.log', self.record_comp_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-commtime.log', self.record_comm_timing, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-acc.log', self.record_accuracy, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-losses.log', self.record_losses, delimiter=',')
    #     np.savetxt(self.saveFolderName + '/lr' + str(self.args.lr) + '-ID' + str(self.id) + '-tacc.log', self.record_trainacc, delimiter=',')
    #     with open(self.saveFolderName + '/ExpDescription', 'w') as f:
    #         f.write(str(self.args) + '\n')
    #         f.write(self.args.description + '\n')


