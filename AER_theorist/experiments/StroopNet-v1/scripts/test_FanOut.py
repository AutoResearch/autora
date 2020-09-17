import torch
from torch.autograd import Variable
from cnnsimple.FanOut import FanOut

# stimulus1 = Variable(torch.FloatTensor([[1], [0]]), requires_grad = False)
# stimulus2 = Variable(torch.FloatTensor([[1], [1]]), requires_grad = False)
#
# model = SimpleNet()
# out = model(stimulus1, stimulus2)
# print(out)

# num_elements = 3
#
# # generate input stimuli
# stimulus = Variable(torch.rand(1, num_elements))   # values for first stimulus are drawn from U(0,1)
#
# # generate labels
# model = FanOut(num_elements)                 # create instance of SimpleNet
# out = model(stimulus)   # get soft-maxed response pattern
#
# for output in out:
#     print(output)

num_elements = 3

# generate input stimuli
stimulus = Variable(torch.rand(1, num_elements))

# generate labels
model = FanOut(num_elements)                 # create instance of FanOut
out = model(stimulus)

print(stimulus)
for output in out:
    print(output)