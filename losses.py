import torch
from torch import tensor
import torch.nn as nn
from itertools import chain

import torchvision

class AngleLoss(nn.Module):

    def __init__(self):
        super(AngleLoss, self).__init__()

    def calculation_error(self,output,target):
        error = torch.acos((torch.dot(output,target))/(torch.norm(
            output)*torch.norm(target)))
        return error

    def forward(self, outputs, targets):
        assert  outputs.ndim == 3
        assert  targets.ndim == 3
        outputs = torch.reshape(outputs,(-1,3))
        targets = torch.reshape(targets,(-1,3))
        # d = arccos(<o,t>/||0||||t||)
        data = list(zip(outputs,targets))
        d = [self.calculation_error(output,target) for output,target in data]
        loss = torch.mean(torch.FloatTensor(d))
        # d_list = [torch.acos(torch.dot(torch.Tensor(output), torch.Tensor(target))/(torch.norm(torch.Tensor(
        #     output))*torch.norm(torch.Tensor(target)))) for output, target in [ _ for _ in zip(outputs, targets)]]
        # loss = [torch.sum(torch.Tensor(d))/len(d) for d in d_list]
        return loss


if __name__ == "__main__":
    from  itertools import chain
    outputs = [
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
    targets = [[[1, 1, 1.1], [2.01, 2, 2], [3, 3.01, 3]],[[1, 1, 1.1], [2.01, 2, 2], [3, 3.01, 3]]]
    # outputs = list(chain(*outputs))
    # targets = list(chain(*targets))
    loss = AngleLoss()
    outputs = torch.FloatTensor(outputs)
    targets = torch.FloatTensor(targets)
    print(len(outputs),len(targets))
    print(loss(outputs,targets))
    
    # print(torch.norm(torch.tensor([0.95772, 0.112233, 0.3441123])))
    # print(math.sqrt(sum([x*x for x in [0.95772, 0.112233, 0.3441123]])))
    
    
    # print()
