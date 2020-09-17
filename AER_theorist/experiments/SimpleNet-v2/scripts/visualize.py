import sys
import cnnsimple.genotypes
from graphviz import Digraph


def plot(genotype, filename, fileFormat='pdf', viewFile=None):
  g = Digraph(
      format=fileFormat,
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("s1", fillcolor='darkseagreen2')
  g.node("s2", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2

  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "s1"
      elif j == 1:
        u = "s2"
      else:
        u = str(j-2)
      v = str(i)
      if op is not 'none':
        g.edge(u, v, label='(' + str(k) + ') ' + op, fillcolor="gray")

  g.node("out", fillcolor='palegoldenrod')
  for i in range(steps):
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

