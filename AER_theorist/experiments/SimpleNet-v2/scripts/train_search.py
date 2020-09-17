import os
import sys
import time
import glob
import numpy as np
import torch
import cnnsimple.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import transforms
import cnnsimple.visualize as viz
import cnnsimple.plot_utils as plotutils
import cnnsimple.genotypes
import matplotlib.pyplot as plt

from torch.autograd import Variable
from cnnsimple.model_search import Network
from cnnsimple.architect import Architect
from cnnsimple.SimpleNet_dataset import SimpleNetDataset


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size') # Edit SM 10/23/19: set from 64 to 16
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency') # Edit SM 10/23/19: set from 50 to 1
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=2, help='num of training epochs') # Edit SM 10/23/19: set from 50 to 5
parser.add_argument('--init_channels', type=int, default=1, help='num of init channels') # Edit SM 10/23/19: set from 16 to 1
# parser.add_argument('--layers', type=int, default=1, help='total number of layers') # Edit SM 10/23/19: set from 7 to 1
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed') # original: 2
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=True, help='use one-step unrolled validation loss') # Edit SM 10/23/19: set to true
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

# create some identifier for log folder
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
# create log folder and copy all current scripts
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

# determine the format for logging specifying the event time and message
log_format = '%(asctime)s %(message)s'
# sets u a logging system in python,
# - the stream is set to to the output console (stdout)
# - report events that occur during normal operation of a program (logging.INFO), i.e. not during debugging
# - use the pre-specified log format with time and message
# - use the corresponding date format
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
# specify handle to file where logging output is stored
fh = logging.FileHandler(os.path.join('exps', args.save, 'log.txt'))
# specify format for logging
fh.setFormatter(logging.Formatter(log_format))
# adds file name to logger
logging.getLogger().addHandler(fh)

# specifies number of categories in the CIFAR data set
CIFAR_CLASSES = 10

debugSearch = True
args.unrolled = False # full optimization (training until w = w*, then architecture update)
architectureUpdatesPerEpoch = 50 # 50
num_datapoints = 5000
num_epochs = 50
architecture_learning_rate = 3e-4
architecture_weight_decay = 1e-3
learning_rate = 0.025
num_nodes = 1
graph_filename = 'model_graph'

args.epochs = num_epochs
args.arch_learning_rate = architecture_learning_rate
args.arch_weight_decay = architecture_weight_decay
args.learning_rate = learning_rate

def main():
  # if not torch.cuda.is_available():
  #   logging.info('no gpu device available')
  #   sys.exit(1)

  # sets seed
  np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  # cudnn.enabled=True
  # torch.cuda.manual_seed(args.seed)
  # log information regarding gpu device and inputs
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # define loss function
  # EDIT 11/04/19 SM: adapting to new SimpleNet data
  # criterion = nn.CrossEntropyLoss()
  criterion = utils.cross_entropy

  # criterion = criterion.cuda()
  # initializes the model given number of channels, output classes and the training criterion
  # EDIT 11/04/19 SM: adapting to new SimpleNet data
  args.init_channels = 1
  CIFAR_CLASSES = 2
  model = Network(CIFAR_CLASSES, criterion, steps=num_nodes)
  # model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # optimizer is standard stochastic gradient decent with some momentum and weight decay
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # EDIT 11/04/19 SM: adapting to new SimpleNet data
  # # transforms data
  # train_transform, valid_transform = utils._data_transforms_cifar10(args)
  # # gets data
  # train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  train_data= SimpleNetDataset(num_datapoints)

  num_train = len(train_data) # number of patterns
  indices = list(range(num_train)) # indices of all patterns
  split = int(np.floor(args.train_portion * num_train)) # size of training set

  # Combines the training set with a sampler, and provides an iterable over the training set
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  # Combines the validation set with a sampler, and provides an iterable over the validation set
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  # generate an architecture of the model
  architect = Architect(model, args)

  if debugSearch:
    # set up plot variables
    numArchEdges = model.alphas_normal.data.shape[0] # number of possible edges between nodes (including input nodes)
    numArchOps = model.alphas_normal.data.shape[1] # number of operations
    ArchOpsLabels = cnnsimple.genotypes.PRIMITIVES

    # function needed to compute BIC & AIC
    softmax = nn.Softmax(dim=1)

    # log variables
    train_error_log = np.empty((args.epochs, 1)) # log training error
    valid_error_log = np.empty((args.epochs, 1)) # log validation error
    BIC_log = np.empty((args.epochs, 1))  # log BIC
    AIC_log = np.empty((args.epochs, 1))  # log AIC
    architecture_weights_log = np.empty((args.epochs, numArchEdges, numArchOps))  # log architecture weights
    train_error_log[:] = np.nan
    valid_error_log[:] = np.nan
    architecture_weights_log[:] = np.nan
    BIC_log[:] = np.nan
    AIC_log[:] = np.nan


    # plot window
    plotWindow = plotutils.DebugWindow(args.epochs, numArchEdges, numArchOps, ArchOpsLabels, fitPlot3D=True)


  for epoch in range(args.epochs):
    # moves the annealing scheduler forward to determine new learning rate
    scheduler.step()
    # get new learning rate
    lr = scheduler.get_lr()[0]
    # log new learning rate
    logging.info('epoch %d lr %e', epoch, lr)

    # returns the genotype of the model
    genotype = model.genotype()
    # logs the genotype of the model
    logging.info('genotype = %s', genotype)

    # prints and log weights of the normal and reduced architecture
    print(F.softmax(model.alphas_normal, dim=-1))
    # print(F.softmax(model.alphas_reduce, dim=-1))

    # training (for one epoch)
    train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, architectureUpdatesPerEpoch)
    # log accuracy on training set
    logging.info('train_acc %f', train_obj)

    # validation (for current epoch)
    valid_obj = infer(valid_queue, model, criterion)
    # log accuracy on validation set
    logging.info('valid_acc %f', valid_obj)

    # stores the model
    utils.save(model, os.path.join(args.save, 'weights.pt'))

    # plot debugging window
    if debugSearch:

      # get model fit data
      # (input1_plot, target_plot, prediction_plot, input_var, target_var) = train_data.sampleModelFit(30, model)
      (input1_plot, input2_plot, target_plot, prediction_plot, input_var, target_var) = train_data.sample_model_fit_2d(30, model)

      # compute BIC and AIC
      soft_prediction = softmax(model(input_var)).data.numpy()
      soft_target = target_var.data.numpy()
      BIC, AIC = utils.compute_BIC_AIC(soft_target, soft_prediction, model)

      # log data
      architecture_weights_log[epoch, :, :] = torch.nn.functional.softmax(model.alphas_normal, dim=-1).data.numpy()
      train_error_log[epoch] = train_obj
      valid_error_log[epoch] = valid_obj
      BIC_log[epoch] = BIC
      AIC_log[epoch] = AIC

      # save model plot
      genotype = model.genotype()
      viz.plot(genotype.normal, graph_filename, fileFormat='png')

      # plot model performance
      plotWindow.update(train_error=train_error_log,
                        valid_error=valid_error_log,
                        weights=architecture_weights_log,
                        model_graph = graph_filename + '.png',
                        BIC = BIC_log,
                        AIC = AIC_log,
                        range_input1=input1_plot,
                        range_input2=input2_plot,
                        range_target = target_plot,
                        range_prediction = prediction_plot,
                        target = soft_target,
                        prediction = soft_prediction
                        )

  viz.plot(genotype.normal, graph_filename)
  (n_params_total, n_params_base) = model.countParameters()
  # valid_acc, valid_obj = infer(valid_queue, model, criterion)


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, architectureUpdatesPerEpoch=1):
  objs = utils.AvgrageMeter() # metric that averages

  objs_log = torch.zeros(5000)

  for step in range(architectureUpdatesPerEpoch):
  # for step, (input, target) in enumerate(train_queue): # for every pattern
    model.train() # Sets the module in training mode
    print(step)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False) #.cuda()
    target_search = Variable(target_search, requires_grad=False) #.cuda(async=True)

    # FIRST STEP: UPDATE ARCHITECTURE (ALPHA)
    # trains the architecture one step (first step: update architecture)
    architect.step(input_search, target_search, lr, optimizer, unrolled=args.unrolled)

  # SECOND STEP: UPDATE MODEL PARAMETERS (W)
    for (input, target) in train_queue:
      # get input and target
      input = Variable(input, requires_grad=False)  # .cuda()
      target = Variable(target, requires_grad=False)  # .cuda(async=True)

      # zero out gradients
      optimizer.zero_grad()
      # compute loss for the model
      logits = model(input)
      loss = criterion(logits, target)
      # update gradients for model
      loss.backward()
      # clips the gradient norm
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      # moves optimizer one step (applies gradients to weights)
      optimizer.step()

      # compute accuracy metrics
      n = input.size(0)
      objs.update(loss.data[0], n)

    objs_log[step] = objs.avg


    if step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('train %03d %e', step, objs.avg)


  return objs.avg


def train_old(train_queue, valid_queue, model, architect, criterion, optimizer, lr, weightUpdatesPerEpoch=1):
  objs = utils.AvgrageMeter() # metric that averages

  objs_log = torch.zeros(5000)


  for step, (input, target) in enumerate(train_queue): # for every pattern
    model.train() # Sets the module in training mode
    print(step)

    # get input and target
    input = Variable(input, requires_grad=False)  # .cuda()
    target = Variable(target, requires_grad=False)  # .cuda(async=True)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False) #.cuda()
    target_search = Variable(target_search, requires_grad=False) #.cuda(async=True)

    # FIRST STEP: UPDATE ARCHITECTURE (ALPHA)
    # trains the architecture one step (first step: update architecture)
    architect.step(input_search, target_search, lr, optimizer, unrolled=args.unrolled, input_train=input, target_train=target)

  # SECOND STEP: UPDATE MODEL PARAMETERS (W)
    for iteration in range(weightUpdatesPerEpoch):

      # zero out gradients
      optimizer.zero_grad()
      # compute loss for the model
      logits = model(input)
      loss = criterion(logits, target)
      # update gradients for model
      loss.backward()
      # clips the gradient norm
      nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
      # moves optimizer one step (applies gradients to weights)
      optimizer.step()

      # compute accuracy metrics
      n = input.size(0)
      objs.update(loss.data[0], n)

    objs_log[step] = objs.avg


    if step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('train %03d %e', step, objs.avg)


  return objs.avg


# computes accuracy for validation set
def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True) #.cuda()
    target = Variable(target, volatile=True) #.cuda(async=True)

    logits = model(input)
    loss = criterion(logits, target)

    # prec1, prec5 = utils.accuracy(logits, target, topk=(1, min(5, logits.shape[1])))
    n = input.size(0)
    objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)

    if step % args.report_freq == 0:
      # logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('valid %03d %e', step, objs.avg)

    # EDIT SM 10/27/19: inserted break for quick run
    # if step == 10:
    #   break

  return objs.avg


if __name__ == '__main__':
  main() 

