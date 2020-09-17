import torch
import numpy as np
from torch.autograd import Variable
from cnnsimple.StroopNet import StroopNet

NUnits = 2
num_patterns = 4

# generate input stimuli
colors = Variable(torch.zeros(num_patterns, NUnits))
words = Variable(torch.zeros(num_patterns, NUnits))
tasks = Variable(torch.zeros(num_patterns, NUnits))

colors.data = torch.FloatTensor([[1, 0], [1, 0], [1, 0], [1, 0]])
words.data = torch.FloatTensor([[1, 0], [0, 1], [1, 0], [0, 1]])
tasks.data = torch.FloatTensor([[1, 0], [1, 0], [0, 1], [0, 1]])

# generate labels
model = StroopNet()                 # create instance of StroopNet
out = model(colors, words, tasks)   # get softmaxed response pattern
print(out)


