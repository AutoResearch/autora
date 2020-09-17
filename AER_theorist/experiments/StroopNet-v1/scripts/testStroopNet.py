import torch
from torch.autograd import Variable
from cnnmod.StroopNet import StroopNet

stimulus = Variable(torch.FloatTensor([[1, 0, 0, 1], [0, 1, 0, 1]]), requires_grad = False)
task = Variable(torch.FloatTensor([[1, 0], [0, 1]]), requires_grad = False)

# stimulus = torch.(1,1,1,4)
# task = torch.empty(1,1,1,2)
# stimulus[0,0,0,:] = torch.tensor([1, 0, 0, 1])
# task[0,0,0,:] = torch.tensor([0, 1])

model = StroopNet()
out = model(stimulus, task)
print(out)