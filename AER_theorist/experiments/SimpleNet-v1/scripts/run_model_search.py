import os
import sys
import time
import glob
import copy
import csv
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F

from torch.autograd import Variable

try:
    import cnnsimple.utils as utils
    import cnnsimple.visualize as viz
    import cnnsimple.plot_utils as plotutils
    import cnnsimple.genotypes
    from cnnsimple.model_search import Network
    from cnnsimple.architect import Architect
    import cnnsimple.model_search_config as cfg
    from cnnsimple.object_of_study import outputTypes
except:
    import utils as utils
    import visualize as viz
    import plot_utils as plotutils
    import genotypes
    from model_search import Network
    from architect import Architect
    import model_search_config as cfg
    from object_of_study import outputTypes



################ PARAMETERIZATION ################

# parse arguments
parser = argparse.ArgumentParser("modelSearch")
parser.add_argument('--slurm_id', type=int, default=0, help='number of slurm array')
parser.add_argument('--object_of_study', type=str, default=cfg.object_of_study, help='name of data generating object of study; available options: \'SimpleNet\' ')
parser.add_argument('--log_version', type=int, default=cfg.log_version, help='log version')
parser.add_argument('--draw_samples', dest='draw_samples', action='store_true', help='generate stochastic samples from study object')
parser.add_argument('--draw_no_samples', dest='draw_samples', action='store_false', help='generate probabilities from study object')
parser.set_defaults(draw_samples=cfg.draw_samples)

args = parser.parse_args()

print('slurm_id: ' + str(args.slurm_id))
print('object_of_study: ' + args.object_of_study)
print('draw_samples: ' + str(args.draw_samples))
print('log_version: ' + str(args.log_version))

# TEMP ARGS
# args.slurm_id = 5

# assign model-specific parameters
args.obj_of_study_class = utils.get_object_of_study(args.object_of_study)
args.obj_of_study = args.obj_of_study_class(num_patterns=cfg.num_data_points, sampling=cfg.draw_samples)
args.inputDim = args.obj_of_study.__get_input_dim__()
args.outputDim = args.obj_of_study.__get_output_dim__()
args.outputType = args.obj_of_study.__get_output_type__()
args.loss = utils.get_loss_function(args.outputType)
args.input_labels = args.obj_of_study.__get_input_labels__()

# Assign default parameters (see model_search_config.py for parameter descriptions)
args.batch_size = cfg.batch_size
args.learning_rate = cfg.learning_rate
args.learning_rate_min = cfg.learning_rate_min
args.momentum = cfg.momentum
args.weight_decay = cfg.weight_decay
args.report_freq = cfg.report_freq
args.gpu = cfg.gpu
args.epochs = cfg.epochs
args.model_path = cfg.model_path
args.exp_folder = cfg.exp_folder
args.seed = cfg.seed
args.grad_clip = cfg.grad_clip
args.train_portion = cfg.train_portion
args.unrolled = cfg.unrolled
args.arch_learning_rate = cfg.arch_learning_rate
args.arch_weight_decay = cfg.arch_weight_decay
args.arch_updates_per_epoch = cfg.arch_updates_per_epoch
args.param_updates_per_epoch = cfg.param_updates_per_epoch
args.n_models_sampled = cfg.n_models_sampled
args.reinitialize_weights = cfg.reinitialize_weights
args.output_file_folder = cfg.output_file_folder
args.debug = cfg.debug
args.num_data_points = cfg.num_data_points
args.graph_filename = cfg.graph_filename
args.model_filename = cfg.model_filename
args.bic_test_size = cfg.bic_test_size
args.sample_amp = cfg.sample_amp
args.classifier_weight_decay = cfg.classifier_weight_decay
args.show_arch_weights = cfg.show_arch_weights

# assign slurm instance
(args.arch_weight_decay_df, args.num_graph_nodes, args.seed) = utils.assign_slurm_instance(slurm_id=args.slurm_id,
                                                                             arch_weight_decay_list=cfg.arch_weight_decay_list,
                                                                             num_node_list=cfg.num_node_list,
                                                                             seed_list=cfg.seed_list)


################ LOGGING ################

# create some identifier for log folder
args.save = '{}-v{}'.format(args.obj_of_study.__get_name__(), str(args.log_version))
# create log folder and copy all current scripts (only for first slurm job)
if args.slurm_id == 0:
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'), parent_folder = args.exp_folder, results_folder = args.output_file_folder)

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
fh = logging.FileHandler(os.path.join(args.exp_folder, args.save, args.output_file_folder, 'log.txt'))
# specify format for logging
fh.setFormatter(logging.Formatter(log_format))
# adds file name to logger
logging.getLogger().addHandler(fh)

# # create path for results file
# output_file_name = utils.create_output_file_name(file_prefix='model_search_' + args.obj_of_study.__get_name__(),
#                                                  log_version = args.log_version,
#                                                  weight_decay=args.arch_weight_decay_df,
#                                                  k=args.num_graph_nodes,
#                                                  seed=args.seed) + '.csv'
# output_file_path = os.path.join(args.exp_folder, args.save, args.output_file_folder, output_file_name)
#

############# MAIN LOOP (ARCHITECTURE SEARCH) ##################

def main():
  # if not torch.cuda.is_available():
  #   logging.info('no gpu device available')
  #   sys.exit(1)

  # sets seed
  np.random.seed(int(args.seed))
  # torch.cuda.set_device(args.gpu)
  # cudnn.benchmark = True
  torch.manual_seed(int(args.seed))
  # cudnn.enabled=True
  # torch.cuda.manual_seed(args.seed)
  # log information regarding gpu device and inputs
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # define loss function
  criterion = utils.get_loss_function(args.obj_of_study.__get_output_type__())

  # criterion = criterion.cuda()
  # initializes the model given number of channels, output classes and the training criterion
  model = Network( args.outputDim, criterion, steps=int(args.num_graph_nodes), n_input_states=int(args.inputDim), classifier_weight_decay=args.classifier_weight_decay)
  # model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  # optimizer is standard stochastic gradient decent with some momentum and weight decay
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  # get data
  train_data = args.obj_of_study_class(num_patterns=args.num_data_points, sampling=args.draw_samples)
  # train_data = args.obj_of_study
  # train_data = SimpleNetDataset(args.num_data_points)
  num_train = len(train_data) # number of patterns
  indices = list(range(num_train)) # indices of all patterns
  split = int(np.floor(args.train_portion * num_train)) # size of training set

  # combine the training set with a sampler, and provides an iterable over the training set
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  # cimbine the validation set with a sampler, and provides an iterable over the validation set
  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  # generate an architecture of the model
  architect = Architect(model, args)

  if args.debug:
    # set up plot variables
    numArchEdges = model.alphas_normal.data.shape[0] # number of possible edges between nodes (including input nodes)
    numArchOps = model.alphas_normal.data.shape[1] # number of operations
    try:
        ArchOpsLabels = cnnsimple.genotypes.PRIMITIVES
    except:
        ArchOpsLabels = genotypes.PRIMITIVES

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


    graph_filename = utils.create_output_file_name(file_prefix=args.graph_filename,
                                                 log_version = args.log_version,
                                                 weight_decay=args.arch_weight_decay_df,
                                                 k=args.num_graph_nodes,
                                                 seed=args.seed)
    graph_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, graph_filename)

    # plot window
    plotWindow = plotutils.DebugWindow(args.epochs, numArchEdges, numArchOps, ArchOpsLabels, fitPlot3D=True, show_arch_weights=args.show_arch_weights)


  for epoch in range(args.epochs): # args.epochs

    # get new learning rate
    lr = scheduler.get_last_lr()[0]
    # log new learning rate
    logging.info('epoch %d lr %e', epoch, lr)

    # returns the genotype of the model
    genotype = model.genotype()
    # # logs the genotype of the model
    logging.info('genotype = %s', genotype)
    #
    # prints and log weights of the normal and reduced architecture
    print(F.softmax(model.alphas_normal, dim=-1))

    # training (for one epoch)
    train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args.arch_updates_per_epoch, args.param_updates_per_epoch)
    # log accuracy on training set
    logging.info('train_acc %f', train_obj)

    # validation (for current epoch)
    valid_obj = infer(valid_queue, model, criterion)
    # log accuracy on validation set
    logging.info('valid_acc %f', valid_obj)

    # moves the annealing scheduler forward to determine new learning rate
    scheduler.step()

    # plot debugging window
    if args.debug:

      # get model fit data
      (input1_plot, input2_plot, target_plot, prediction_plot, input_var, target_var) = train_data.sample_model_fit_2d(30, model)

      # compute BIC and AIC
      soft_target = target_var.data.numpy()
      if args.outputType is outputTypes.PROBABILITY_DISTRIBUTION or args.outputType is outputTypes.CLASS:
          soft_prediction = softmax(model(input_var)).data.numpy()
          BIC, AIC = utils.compute_BIC_AIC(soft_target, soft_prediction, model)
      elif args.outputType is outputTypes.PROBABILITY:
          soft_prediction = model(input_var).data.numpy()
          BIC, AIC = utils.compute_BIC_AIC(soft_target, soft_prediction, model)
      else:
          soft_prediction = model(input_var).data.numpy()
          BIC = 0
          AIC  = 0


      # log data
      architecture_weights_log[epoch, :, :] = torch.nn.functional.softmax(model.alphas_normal, dim=-1).data.numpy()
      train_error_log[epoch] = train_obj
      valid_error_log[epoch] = valid_obj
      BIC_log[epoch] = BIC
      AIC_log[epoch] = AIC

      # save model plot
      genotype = model.genotype()

      viz.plot(genotype.normal, args.num_graph_nodes, graph_filepath, fileFormat='png', input_labels=args.input_labels)

      # plot model performance
      plotWindow.update(train_error=train_error_log,
                        valid_error=valid_error_log,
                        weights=architecture_weights_log,
                        model_graph = graph_filepath + '.png',
                        BIC = BIC_log,
                        AIC = AIC_log,
                        range_input1=input1_plot,
                        range_input2=input2_plot,
                        range_target = target_plot,
                        range_prediction = prediction_plot,
                        target = soft_target,
                        prediction = soft_prediction
                        )

  # # save image of final genotype
  # if(args.debug):
  #   viz.plot(genotype.normal, args.num_graph_nodes, graph_filepath, input_labels=args.input_labels)
  #
  # # stores the model and architecture
  # model_filename = utils.create_output_file_name(file_prefix='model_weights',
  #                                                log_version = args.log_version,
  #                                                weight_decay=args.arch_weight_decay_df,
  #                                                k=args.num_graph_nodes,
  #                                                seed=args.seed)
  #
  # arch_filename = utils.create_output_file_name(file_prefix='arch_weights',
  #                                                log_version=args.log_version,
  #                                                weight_decay=args.arch_weight_decay_df,
  #                                                k=args.num_graph_nodes,
  #                                                seed=args.seed)
  #
  # model_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, model_filename + '.pt')
  # arch_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, arch_filename + '.pt')
  #
  # utils.save(model, model_filepath)
  # torch.save(model.alphas_normal, arch_filepath)
  #
  # # generate validation set for computing BIC and AIC
  # bic_valid_queue = torch.utils.data.DataLoader(
  #     train_data, batch_size=args.bic_test_size,
  #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
  #     pin_memory=True, num_workers=2)
  #
  # evaluate_architectures(train_data, train_queue, valid_queue, bic_valid_queue, model, criterion)



def evaluate_architectures(train_data, train_queue, valid_queue, bic_valid_queue, model, criterion):

  criterion_loss_log = list()
  BIC_log = list()
  AIC_log = list()
  model_name_log = list()

  # generate general model file name
  model_filename_gen = utils.create_output_file_name(file_prefix='model_weights',
                                                     log_version=args.log_version,
                                                     weight_decay=args.arch_weight_decay_df,
                                                     k=args.num_graph_nodes,
                                                     seed=args.seed)

  arch_filename_gen = utils.create_output_file_name(file_prefix='arch_weights',
                                                     log_version=args.log_version,
                                                     weight_decay=args.arch_weight_decay_df,
                                                     k=args.num_graph_nodes,
                                                     seed=args.seed)

  # generate test set for computing BIC and AIC
  input_full, target_full = next(iter(bic_valid_queue))

  # subsample models and retrain
  sampled_weights = list()
  for sample_id in range(args.n_models_sampled):

      logging.info('architecture evaluation for sampled model %d / %d', sample_id+1, args.n_models_sampled)

      # sample architecture weights
      found_weights = False
      if(sample_id == 0):
          candidate_weights = model.max_alphas_normal()
          found_weights = True
      else:
          candidate_weights = model.sample_alphas_normal(args.sample_amp)

      while found_weights is False:
            weights_are_novel = True
            for logged_weights in sampled_weights:
                if torch.eq(logged_weights, candidate_weights).all() is True:
                    weights_are_novel = False
            if weights_are_novel:
                novel_weights = candidate_weights
                found_weights = True
            else:
                candidate_weights = model.sample_alphas_normal()


      # store sampled architecture weights
      sampled_weights.append(candidate_weights)

      # reinitialize weights if desired
      if args.reinitialize_weights:
          new_model = Network(args.outputDim, criterion, steps=int(args.num_graph_nodes), n_input_states=int(args.inputDim), classifier_weight_decay=args.classifier_weight_decay)
      else:
          new_model = copy.deepcopy(model)

      new_model.fix_architecture(True, candidate_weights)

      # optimizer is standard stochastic gradient decent with some momentum and weight decay
      optimizer = torch.optim.SGD(
          new_model.parameters(),
          args.learning_rate,
          momentum=args.momentum,
          weight_decay=args.weight_decay)

      # Set the learning rate of each parameter group using a cosine annealing schedule (model optimization)
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
          optimizer, float(args.epochs), eta_min=args.learning_rate_min)

      # train model
      for epoch in range(args.epochs):
          # moves the annealing scheduler forward to determine new learning rate
          scheduler.step()
          # get new learning rate
          lr = scheduler.get_lr()[0]

          new_model.train()  # Sets the module in training mode

          for param_step in range(args.param_updates_per_epoch):
              # get input and target
              input_search, target_search = next(iter(train_queue))
              input = Variable(input_search, requires_grad=False)  # .cuda()
              target = Variable(target_search, requires_grad=False)  # .cuda(async=True)

              input, target = format_input_target(input, target, criterion)

              # zero out gradients
              optimizer.zero_grad()
              # compute loss for the model
              logits = new_model(input)
              loss = criterion(logits, target)
              # update gradients for model
              loss.backward()
              # clips the gradient norm
              nn.utils.clip_grad_norm(new_model.parameters(), args.grad_clip)
              # moves optimizer one step (applies gradients to weights)
              optimizer.step()
              # applies weight decay to classifier weights
              model.apply_weight_decay_to_classifier(lr)


          # if in debug mode, print loss during architecture evaluation
          logging.info('epoch %d', epoch)
          if(args.debug):
               criterion_loss = infer(valid_queue, new_model, criterion)
               (BIC, AIC) = get_BIC_AIC(input_full, target_full, new_model)
               logging.info('criterion loss %f, BIC %f, AIC %f', criterion_loss, BIC, AIC)

      # evaluate model
      criterion_loss = infer(valid_queue, new_model, criterion, silent = True)
      (BIC, AIC) = get_BIC_AIC(input_full, target_full, new_model)

      criterion_loss_log.append(criterion_loss)
      BIC_log.append(BIC)
      AIC_log.append(AIC)

      # get model name
      model_filename = model_filename_gen + '_sample' + str(sample_id)
      arch_filename = arch_filename_gen + '_sample' + str(sample_id)
      model_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, model_filename + '.pt')
      arch_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, arch_filename + '.pt')
      model_graph_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, model_filename)
      model_name_log.append(model_filename)
      genotype = new_model.genotype()

      # save model
      utils.save(new_model, model_filepath)
      torch.save(new_model.alphas_normal, arch_filepath)
      print('SAVING MODEL WEIGHTS TO PATH: ' + model_filepath)
      viz.plot(genotype.normal, args.num_graph_nodes, model_graph_filepath, viewFile=False, input_labels=args.input_labels)
      print('SAVING MODEL GRAPH TO PATH: ' + model_graph_filepath)
      print('SAVING ARCHITECTURE WEIGHTS TO PATH: ' + arch_filepath)

  # get name for csv log file
  model_filename_csv = model_filename_gen + '.csv'
  model_filepath = os.path.join(args.exp_folder, args.save, args.output_file_folder, model_filename_csv)

  # save csv file
  rows = zip(model_name_log, criterion_loss_log, BIC_log, AIC_log)
  with open(model_filepath, "w") as f:
      writer = csv.writer(f)
      for row in rows:
          writer.writerow(row)

def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, arch_updates_per_epoch=1, param_updates_per_epoch = 1):
  objs = utils.AvgrageMeter() # metric that averages

  objs_log = torch.zeros(train_queue.dataset._num_patterns)

  for arch_step in range(arch_updates_per_epoch):
  # for step, (input, target) in enumerate(train_queue): # for every pattern
    model.train() # Sets the module in training mode
    print(arch_step)

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue))
    input_search = Variable(input_search, requires_grad=False) #.cuda()
    target_search = Variable(target_search, requires_grad=False) #.cuda(async=True)

    input_search, target_search = format_input_target(input_search, target_search, criterion)

    # FIRST STEP: UPDATE ARCHITECTURE (ALPHA)
    # trains the architecture one step (first step: update architecture)
    architect.step(input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    # SECOND STEP: UPDATE MODEL PARAMETERS (W)
    for param_step in range(param_updates_per_epoch):

      # get input and target
      input_search, target_search = next(iter(train_queue))
      input = Variable(input_search, requires_grad=False)  # .cuda()
      target = Variable(target_search, requires_grad=False)  # .cuda(async=True)

      input, target = format_input_target(input, target, criterion)

      # zero out gradients
      optimizer.zero_grad()
      # compute loss for the model
      logits = model(input)
      loss = criterion(logits, target)
      # update gradients for model
      loss.backward()
      # clips the gradient norm
      nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
      # moves optimizer one step (applies gradients to weights)
      optimizer.step()
      # applies weight decay to classifier weights
      model.apply_weight_decay_to_classifier(lr)

      # compute accuracy metrics
      n = input.size(0)
      objs.update(loss.data, n)

    objs_log[arch_step] = objs.avg


    if arch_step % args.report_freq == 0:
      # logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
      logging.info('train %03d %e', arch_step, objs.avg)


  return objs.avg


def format_input_target(input, target, criterion):

    if isinstance(criterion, nn.CrossEntropyLoss):
        target = target.squeeze()

    return (input, target)


def get_BIC_AIC(input_full, target_full, model):

    input_full = Variable(input_full, requires_grad=False)  # .cuda()
    target_full = Variable(target_full, requires_grad=False)  # .cuda(async=True)

    # function needed to compute BIC & AIC
    softmax = nn.Softmax(dim=1)

    if args.outputType is outputTypes.PROBABILITY_DISTRIBUTION or args.outputType is outputTypes.CLASS:
        soft_prediction = softmax(model(input_full)).data.numpy()
        BIC, AIC = utils.compute_BIC_AIC(target_full, soft_prediction, model)
        BIC = BIC.flatten()[0].data.numpy()
        AIC = AIC.data.numpy()
    elif args.outputType is outputTypes.PROBABILITY:
        soft_prediction = model(input_full).data.numpy()
        BIC, AIC = utils.compute_BIC_AIC(target_full, soft_prediction, model)
        BIC = BIC.flatten()[0].data.numpy()
        AIC = AIC.data.numpy()
    else:
        BIC = 0
        AIC = 0

    return (BIC, AIC)

# computes accuracy for validation set
def infer(valid_queue, model, criterion, silent = False):
  objs = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, requires_grad=True) #.cuda()
    target = Variable(target, requires_grad=True) #.cuda(async=True)

    input, target = format_input_target(input, target, criterion)

    logits = model(input)
    loss = criterion(logits, target)

    n = input.size(0)
    objs.update(loss.data, n)

    if silent is False:
        if step % args.report_freq == 0:
          logging.info('valid %03d %e', step, objs.avg)

  return objs.avg


if __name__ == '__main__':
  main()