import os
import numpy as np
import torch
import torch.nn as nn
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
from cnnsimple.SimpleNet_dataset import SimpleNetDataset
from cnnsimple.object_of_study import outputTypes

def create_output_file_name(file_prefix, log_version = None, weight_decay = None, k = None, seed = None):

    output_str = file_prefix

    if log_version is not None:
        output_str += '_v_' + str(log_version)

    if weight_decay is not None:
        output_str += '_wd_' + str(weight_decay)

    if k is not None:
        output_str += '_k_' + str(k)

    if k is not None:
        output_str += '_s_' + str(seed)

    return output_str


def assign_slurm_instance(slurm_id, arch_weight_decay_list = None, num_node_list = None, seed_list = None):

    seed_id = np.floor(slurm_id / (len(num_node_list) * len (arch_weight_decay_list))) % len(seed_list)
    k_id = np.floor(slurm_id / (len (arch_weight_decay_list))) % len(num_node_list)
    weight_decay_id = slurm_id % len(arch_weight_decay_list)

    return (arch_weight_decay_list[int(weight_decay_id)], int(num_node_list[int(k_id)]), int(seed_list[int(seed_id)]))

def get_object_of_study(studyObject):

    dataSets = {
        'SimpleNet':  SimpleNetDataset
    }

    return dataSets.get(studyObject, SimpleNetDataset)


def get_loss_function(outputType):

    dataSets = {
        outputTypes.REAL:  nn.MSELoss(),
        outputTypes.PROBABILITY: nn.MSELoss(),
        outputTypes.PROBABILITY_DISTRIBUTION: cross_entropy,
        outputTypes.CLASS: nn.CrossEntropyLoss(),
    }

    return dataSets.get(outputType, nn.MSELoss)


def compute_BIC_AIC(soft_targets, soft_prediction, model):

    lik = np.sum(np.multiply(soft_prediction, soft_targets), axis=1)    # likelihood of data given model
    llik = np.sum(np.log(lik))                                          # log likelihood
    n = len(lik)                                                        # number of data points
    k,_ = model.countParameters()                                         # for most likely architecture

    BIC = np.log(n) * k - 2 * llik

    AIC = 2 * k - 2 * llik

    return BIC, AIC


def cross_entropy(pred, soft_targets):
    # assuming pred and soft_targets are both Variables with shape (batchsize, num_of_classes),
    # each row of pred is predicted logits and each row of soft_targets is a discrete distribution.
    logsoftmax = nn.LogSoftmax(dim=1)
    return torch.mean(torch.sum(- soft_targets * logsoftmax(pred), 1))

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  if is_best:
    best_filename = os.path.join(save, 'model_best.pth.tar')
    shutil.copyfile(filename, best_filename)


def save(model, model_path, exp_folder = None):
  if exp_folder is not None:
    os.chdir('exps')  # Edit SM 10/23/19: use local experiment directory
  torch.save(model.state_dict(), model_path)
  if exp_folder is not None:
    os.chdir('..')  # Edit SM 10/23/19: use local experiment directory

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x


def create_exp_dir(path, scripts_to_save=None, parent_folder = 'exps', results_folder = None):
  os.chdir(parent_folder) # Edit SM 10/23/19: use local experiment directory
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if results_folder is not None:
    try:
        os.mkdir(os.path.join(path, results_folder))
    except OSError:
        pass

  if scripts_to_save is not None:
    try:
        os.mkdir(os.path.join(path, 'scripts'))
    except OSError:
        pass
    os.chdir('..') # Edit SM 10/23/19: use local experiment directory
    for script in scripts_to_save:
      dst_file = os.path.join(parent_folder, path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

