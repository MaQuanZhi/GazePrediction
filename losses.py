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
        [[0.21621717167064639,0.19047621894625918,0.9575849542942302],
        [0.2162010641522083,0.19083801668519412,0.957516554032939],
        [0.21549008832928146,0.1917042129076262,0.9575037945539986],
        [0.21547878232473153,0.1929996013651404,0.9572460750719993],
        [0.21547878232473153,0.1929996013651404,0.9572460750719993]],
        [[0.21621717167064639,0.19047621894625918,0.9575849542942302],
        [0.2162010641522083,0.19083801668519412,0.957516554032939],
        [0.21549008832928146,0.1917042129076262,0.9575037945539986],
        [0.21547878232473153,0.1929996013651404,0.9572460750719993],
        [0.21547878232473153,0.1929996013651404,0.9572460750719993]]
        ]
    targets = [
        [[0.21601863387030387,0.19299833227844454,0.957124648913895],
        [0.21760697189425648,0.19284153420306258,0.956796398649797],
        [0.21603570941374245,0.19261908409833725,0.957197190081134],
        [0.21613678341178508,0.19261451081193748,0.9571752927656904],
        [0.2161354498051421,0.19185654213134007,0.9573278093625684]],
        [[0.21601863387030387,0.19299833227844454,0.957124648913895],
        [0.21760697189425648,0.19284153420306258,0.956796398649797],
        [0.21603570941374245,0.19261908409833725,0.957197190081134],
        [0.21613678341178508,0.19261451081193748,0.9571752927656904],
        [0.2161354498051421,0.19185654213134007,0.9573278093625684]]
        ]
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
