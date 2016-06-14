import sys
import pandas as pd
from copy import deepcopy
from optparse import OptionParser

sys.path.append('../')
from mcmc import *
from parallel import *

sys.path.append('../Prior')
from fit_prior import read_prior_par

import iodata

# -----------------------------------------------------------------------------
def parse_options():
    """Parse command-line arguments.

    """
    parser = OptionParser(usage='usage: %prog [options] DATASET')
    parser.add_option("-p", "--priorpar", dest="pparfile", default=None,
                      help="Use priors from this file (default: no priors)")
    return parser

# -----------------------------------------------------------------------------
NS = 1000000
THIN = 1

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # Arguments
    parser = parse_options()
    opt, args = parser.parse_args()
    dset = args[0]

    VARS = iodata.XVARS[dset]
    Y = iodata.YLABS[dset]

    # Temperatures
    Ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Read the data
    inFileName = '%s/data/%s' % (dset, iodata.FNAMES[dset])
    data, x, y = iodata.read_data(
        dset, ylabel=Y, xlabels=VARS, in_fname=inFileName,
    )
    if opt.pparfile != None:
        prior_par = read_prior_par(opt.pparfile)

    # Get the output file ready
    try:
        outfname = '%s/sample_charac_parallel__%s' % (
            dset, opt.pparfile.split('/')[-1]
        )
        npar = outfname[outfname.find('.np') + 3:]
        npar = int(npar[:npar.find('.')])
    except AttributeError:
        outfname = '%s/sample_charac_parallel' % (dset)
    outf = open(outfname, 'w')
    print >> outf, '#', prior_par
    
    # Initialize
    p = Parallel(
        Ts,
        variables=VARS,
        parameters=['a%d' % i for i in range(npar)],
        x=x, y=y,
        prior_par=prior_par,
    )
    print >> outf, '# Variables: ', p.t1.variables
    print >> outf, '# Parameters:', p.t1.parameters, '\n#'

    # Burnin
    #for i in range(100):
    #    p.mcmc_step()

    # MCMC
    vprob = dict([(v, 0) for v in VARS])
    sizeprob = dict([(s, 0) for s in range(p.t1.max_size+1)])
    nparprob = dict([(n, 0) for n in range(npar+1)])
    for i in range(NS):
        print >> sys.stdout, i+1
        print >> outf, ' || '.join(
            [str(tmp) for tmp in [i+1, p.t1.E, p.t1.get_energy()[0],
                                  p.t1.EB*2, p.t1.bic,
                                  p.t1.size, p.t1.canonical(),
                                  p.t1, p.t1.par_values]]
        )
        outf.flush()
        for kk in range(THIN):
            p.mcmc_step()
            p.tree_swap()
        for v in VARS:
            if v in [n.value for n in p.t1.ets[0]]:
                vprob[v] += 1
        nparprob[len(set([n.value for n in p.t1.ets[0]
                          if n.value in p.t1.parameters]))] += 1
        sizeprob[p.t1.size] += 1
        
    # Report
    print >> outf, '\n# formula size' 
    for s in sizeprob:
        print >> outf, s, sizeprob[s] / float(NS)

    print >> outf, '\n# variable use' 
    for v in vprob:
        print >> outf, v, vprob[v] / float(NS)

    print >> outf, '\n# parameters' 
    for n in nparprob:
        print >> outf, n, nparprob[n] / float(NS)

    # Done
    outf.close()
