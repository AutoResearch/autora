import sys
from random import random
from subprocess import call
from pprint import pprint

def read_trace(inFileName):
    with open(inFileName) as inf:
        tracedict = dict([(line.strip().split(' || ')[6],
                           line.strip().split(' || '))
                          for line in inf.readlines()
                          if not line.startswith('#') and
                          ' || ' in line])
        trace = [l for c, l in tracedict.items()]
    return trace

def get_top(trace, n=10, col=1, reverse=False):
    dec_trace = [(float(line[col]), random(), line) for line in trace]
    dec_trace.sort()
    if reverse:
        dec_trace.reverse()
    return [line for s, tb, line in dec_trace[:n]]

def parse_model_properties(inFileName='.model.tmp'):
    data = {}
    with open(inFileName) as inf:
        for line in [l for l in inf.readlines() if not l.startswith('--')]:
            k = line.strip().split()[0][:-1]
            v = ' '.join(line.strip().split()[1:])
            data[k] = v
    return data

def main(inFileName):
    dset = inFileName.split('/')[0]
    priorFileName = inFileName[inFileName.find('__')+2 :]

    trace = read_trace(inFileName)
    top = get_top(trace, col=1, n=10)
    model_prop = {}
    for i in range(len(top)):
        print '>>>>> MODEL', i+1, top[i][1], top[i][3], top[i][6]
        cm = 'python model_charac.py -p \"../Prior/%s\" -v \"%s\" %s \"%s\"' % (
            priorFileName,
            top[i][8],
            dset,
            top[i][7],
        )
        call(cm + '> .model.tmp', shell=True)
        model_prop[i] = parse_model_properties()
        call('cat .model.tmp', shell=True)

    for i in range(len(model_prop)):
        try:
            print ' & '.join((
                '$%s$' % model_prop[i]['LaTeX'],
                '%.3f' % float(model_prop[i]['-log(F)'].split(', ')[0][1:]),
                '%.1f' % float(model_prop[i]['BIC']),
                model_prop[i]['LOO-MAE'].replace('+-', ' $\\pm $'),
            )) + ' \\\\'
        except KeyError:
            pass

if __name__ == '__main__':
    inFileName = sys.argv[1]
    main(inFileName)
