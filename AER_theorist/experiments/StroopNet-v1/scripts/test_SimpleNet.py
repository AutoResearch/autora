import torch
from torch.autograd import Variable
from cnnsimple.SimpleNet import SimpleNet

# stimulus1 = Variable(torch.FloatTensor([[1], [0]]), requires_grad = False)
# stimulus2 = Variable(torch.FloatTensor([[1], [1]]), requires_grad = False)
#
# model = SimpleNet()
# out = model(stimulus1, stimulus2)
# print(out)

num_patterns = 10

# generate input stimuli
stimulus1 = Variable(torch.rand(num_patterns, 1))   # values for first stimulus are drawn from U(0,1)
stimulus2 = Variable(torch.rand(num_patterns, 1))   # values for second stimulus are drawn from U(0,1)

stimulus1 = Variable(torch.ones(num_patterns, 1) * 1)
stimulus2 = Variable(torch.ones(num_patterns, 1) * 0)

# generate labels
model = SimpleNet()                 # create instance of SimpleNet
out = model(stimulus1, stimulus2)   # get soft-maxed response pattern
uniformSample = Variable(torch.rand(num_patterns, 1))  # get uniform sample for each response
finalResponse = Variable(torch.zeros(num_patterns, 1))
for i, (response, sample) in enumerate(zip(out, uniformSample)): # determine final response
    finalResponse[i] = 1 if (response[0] > sample).all() == 1 else 2


print(out)
print(uniformSample)
print(finalResponse)

