import sys
from graphviz import Digraph

from AER_theorist.darts.operations import *

def plot(genotype, filename, fileFormat='pdf', viewFile=None, full_label=False, param_list=(), input_labels=()):

  decimals_to_display = 2
  g = Digraph(
      format=fileFormat,
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  for input_node in input_labels:
    g.node(input_node, fillcolor='darkseagreen2')
  # assert len(genotype) % 2 == 0

  # determine number of steps (intermediate nodes)
  steps = 0
  for op, j in genotype:
    if j == 0:
      steps += 1

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  n = len(input_labels)
  start = 0
  for i in range(steps):
    end = start + n
    # for k in [2*i, 2*i + 1]:
    for k in range(start, end): # adapted this iteration from get_genotype() in model_search.py
      op, j = genotype[k]
      if j < len(input_labels):
        u = input_labels[j]
      else:
        u = str(j-2)
      v = str(i)
      if op is not 'none':
        op_label = op
        if full_label:
          params = param_list[k]
          op_label = get_operation_label(op, params, decimals=decimals_to_display)
          g.edge(u, v, label=op_label, fillcolor="gray")
        else:
          g.edge(u, v, label='(' + str(j+start) + ') ' + op_label, fillcolor="gray") # '(' + str(k) + ') '
    start = end
    n += 1

  g.node("out", fillcolor='palegoldenrod')
  for i in range(steps):
    if full_label:
      params = param_list[-1]
      op_label = get_operation_label('classifier', params, decimals=decimals_to_display)
      g.edge(str(i), "out", label=op_label, fillcolor="gray")
    else:
      g.edge(str(i), "out", fillcolor="gray")

  if viewFile is None:
    if(fileFormat == 'pdf'):
      viewFile = True
    else:
      viewFile = False

  g.render(filename, view=viewFile)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
    sys.exit(1)

  genotype_name = sys.argv[1]
  try:
    genotype = eval('genotypes.{}'.format(genotype_name))
  except AttributeError:
    print("{} is not specified in genotypes.py".format(genotype_name)) 
    sys.exit(1)

  plot(genotype.normal, "normal")
  plot(genotype.reduce, "reduction")

