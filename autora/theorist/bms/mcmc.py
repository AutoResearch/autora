"""
A Markov-Chain Monte-Carlo module.

Module constants:
    `get_ops()`:
        A dictionary of accepted operations: `{operation_name: offspring}`

        `operation_name`: the operation name, e.g. 'sin' for the sinusoid function

        `offspring`: the number of arguments the function requires.

        For instance, `get_ops() = {"sin": 1, "**": 2 }` means for
        `sin` the function call looks like `sin(x1)` whereas for
        the exponentiation operator `**`, the function call looks like `x1 ** x2`
"""

import json
import logging
import sys
from copy import deepcopy
from itertools import permutations, product
from random import choice, random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from sympy import lambdify, latex, log, sympify

from .prior import get_priors, relu

_logger = logging.getLogger(__name__)


class Node:
    """
    Object that holds algebraic term. This could be a function, variable, or parameter.

    Attributes:
        order: number of children nodes this term has
                e.g. cos(x) has one child, whereas add(x,y) has two children
    """

    def __init__(self, value, parent=None, offspring=[]):
        """
        Initialises the node object.

        Arguments:
            parent: parent node - unless this node is the root, this will be whichever node contains
                    the function this node's term is most immediately nested within
                    e.g. f(x) is the parent of g(x) in f(g(x))
            offspring: list of child nodes
            value: the specific term held by this node
        """
        self.parent: Node = parent
        self.offspring: List[Node] = offspring
        self.value: str = value
        self.order: int = len(self.offspring)

    def pr(self, show_pow=False):
        """
        Converts expression in readable form

        Returns: String
        """
        if self.offspring == []:
            return "%s" % self.value
        elif len(self.offspring) == 2:
            return "(%s %s %s)" % (
                self.offspring[0].pr(show_pow=show_pow),
                self.value,
                self.offspring[1].pr(show_pow=show_pow),
            )
        else:
            if show_pow:
                return "%s(%s)" % (
                    self.value,
                    ",".join([o.pr(show_pow=show_pow) for o in self.offspring]),
                )
            else:
                if self.value == "pow2":
                    return "(%s ** 2)" % (self.offspring[0].pr(show_pow=show_pow))
                elif self.value == "pow3":
                    return "(%s ** 3)" % (self.offspring[0].pr(show_pow=show_pow))
                else:
                    return "%s(%s)" % (
                        self.value,
                        ",".join([o.pr(show_pow=show_pow) for o in self.offspring]),
                    )


class Tree:
    """
    Object that manages the model equation. It contains the root node, which in turn iteratively
    holds children nodes. Collectively this represents the model equation tree

    Attributes:
        root: the root node of the equation tree
        parameters: the settable parameters for this trees model search
        op_orders: order of each function within the ops
        nops: number of operations of each type
        move_types: possible combinations of function nesting
        ets: possible elementary equation trees
        dist_par: distinct parameters used
        nodes: nodes of the tree (operations and leaves)
        et_space: space of all possible leaves and elementary trees
        rr_space: space of all possible root replacement trees
        num_rr: number of possible root replacement trees
        x: independent variable data
        y: depedent variable data
        par_values: The values of the model parameters (one set of values for each dataset)
        fit_par: past successful parameter fittings
        sse: sum of squared errors (measure of goodness of fit)
        bic: bayesian information criterion (measure of goodness of fit)
        E: total energy of model
        EB: fraction of energy derived from bic score of model
        EP: fraction of energy derived from model given prior
        representative: representative tree for each canonical formula
    """

    prior, ops = get_priors()

    def __init__(
        self,
        ops=ops,
        variables=["x"],
        parameters=["a"],
        prior_par=prior,
        x=None,
        y=None,
        BT=1.0,
        PT=1.0,
        max_size=50,
        root_value=None,
    ):
        """
        Initialises the tree object

        Args:
            ops: allowed operations to compose equation
            variables: dependent variable names
            parameters: parameters that can be used to better fit the equation to the data
            prior_par: hyperparameter values over operations within ops
            x: dependent variables
            y: independent variables
            BT: BIC value corresponding to equation
            PT: prior temperature
            max_size: maximum size of tree (maximum number of nodes)
            root_value: algebraic term held at root of equation
        """
        # The variables and parameters
        self.variables = variables
        self.parameters = [
            p if p.startswith("_") and p.endswith("_") else "_%s_" % p
            for p in parameters
        ]
        # The root
        if root_value is None:
            self.root = Node(
                choice(self.variables + self.parameters), offspring=[], parent=None
            )
        else:
            self.root = Node(root_value, offspring=[], parent=None)
        # The possible operations
        self.ops = ops
        # The possible orders of the operations, move types, and move
        # type probabilities
        self.op_orders = list(set([0] + [n for n in list(ops.values())]))
        self.move_types = [p for p in permutations(self.op_orders, 2)]
        # Elementary trees (including leaves), indexed by order
        self.ets = dict([(o, []) for o in self.op_orders])
        self.ets[0] = [self.root]
        # Distinct parameters used
        self.dist_par = list(
            set([n.value for n in self.ets[0] if n.value in self.parameters])
        )
        self.n_dist_par = len(self.dist_par)
        # Nodes of the tree (operations + leaves)
        self.nodes = [self.root]
        # Tree size and other properties of the model
        self.size = 1
        self.max_size = max_size
        # Space of all possible leaves and elementary trees
        # (dict. indexed by order)
        self.et_space = self.build_et_space()
        # Space of all possible root replacement trees
        self.rr_space = self.build_rr_space()
        self.num_rr = len(self.rr_space)
        # Number of operations of each type
        self.nops = dict([[o, 0] for o in ops])
        # The parameters of the prior probability (default: 5 everywhere)
        if prior_par == {}:
            self.prior_par = dict([("Nopi_%s" % t, 10.0) for t in self.ops])
        else:
            self.prior_par = prior_par
        # The datasets
        if x is None:
            self.x = {"d0": pd.DataFrame()}
            self.y = {"d0": pd.Series(dtype=float)}
        elif isinstance(x, pd.DataFrame):
            self.x = {"d0": x}
            self.y = {"d0": y}
        elif isinstance(x, dict):
            self.x = x
            if y is None:
                self.y = dict([(ds, pd.Series(dtype=float)) for ds in self.x])
            else:
                self.y = y
        else:
            raise TypeError("x must be either a dict or a pandas.DataFrame")
        # The values of the model parameters (one set of values for each dataset)
        self.par_values = dict(
            [(ds, deepcopy(dict([(p, 1.0) for p in self.parameters]))) for ds in self.x]
        )
        # BIC and prior temperature
        self.BT = float(BT)
        self.PT = float(PT)
        # For fast fitting, we save past successful fits to this formula
        self.fit_par = {}
        # Goodness of fit measures
        self.sse = self.get_sse()
        self.bic = self.get_bic()
        self.E, self.EB, self.EP = self.get_energy()
        # To control formula degeneracy (i.e. different trees that
        # correspond to the same cannoninal formula), we store the
        # representative tree for each canonical formula
        self.representative = {}
        self.representative[self.canonical()] = (
            str(self),
            self.E,
            deepcopy(self.par_values),
        )
        # Done
        return

    # -------------------------------------------------------------------------
    def __repr__(self):
        """
        Updates tree's internal representation

        Returns: root node representation

        """
        return self.root.pr()

    # -------------------------------------------------------------------------
    def pr(self, show_pow=True):
        """
        Returns readable representation of tree's root node

        Returns: root node representation

        """
        return self.root.pr(show_pow=show_pow)

    # -------------------------------------------------------------------------
    def canonical(self, verbose=False):
        """
        Provides canonical form of tree's equation so that functionally equivalent trees
        are made into structurally equivalent trees

        Return: canonical form of a tree
        """
        try:
            cansp = sympify(str(self).replace(" ", ""))
            can = str(cansp)
            ps = list([str(s) for s in cansp.free_symbols])
            positions = []
            for p in ps:
                if p.startswith("_") and p.endswith("_"):
                    positions.append((can.find(p), p))
            positions.sort()
            pcount = 1
            for pos, p in positions:
                can = can.replace(p, "c%d" % pcount)
                pcount += 1
        except SyntaxError:
            if verbose:
                print(
                    "WARNING: Could not get canonical form for",
                    str(self),
                    "(using full form!)",
                    file=sys.stderr,
                )
            can = str(self)
        return can.replace(" ", "")

    # -------------------------------------------------------------------------
    def latex(self):
        """
        translate equation into latex

        Returns: canonical latex form of equation
        """
        return latex(sympify(self.canonical()))

    # -------------------------------------------------------------------------
    def build_et_space(self):
        """
        Build the space of possible elementary trees,
        which is a dictionary indexed by the order of the elementary tree

        Returns: space of elementary trees
        """
        et_space = dict([(o, []) for o in self.op_orders])
        et_space[0] = [[x, []] for x in self.variables + self.parameters]
        for op, noff in list(self.ops.items()):
            for vs in product(et_space[0], repeat=noff):
                et_space[noff].append([op, [v[0] for v in vs]])
        return et_space

    # -------------------------------------------------------------------------
    def build_rr_space(self):
        """
        Build the space of possible trees for the root replacement move

        Returns: space of possible root replacements
        """
        rr_space = []
        for op, noff in list(self.ops.items()):
            if noff == 1:
                rr_space.append([op, []])
            else:
                for vs in product(self.et_space[0], repeat=(noff - 1)):
                    rr_space.append([op, [v[0] for v in vs]])
        return rr_space

    # -------------------------------------------------------------------------
    def replace_root(self, rr=None, update_gof=True, verbose=False):
        """
        Replace the root with a "root replacement" rr (if provided;
        otherwise choose one at random from self.rr_space)

        Returns: new root (if move was possible) or None (otherwise)
        """
        # If no RR is provided, randomly choose one
        if rr is None:
            rr = choice(self.rr_space)
        # Return None if the replacement is too big
        if (self.size + self.ops[rr[0]]) > self.max_size:
            return None
        # Create the new root and replace existing root
        newRoot = Node(rr[0], offspring=[], parent=None)
        newRoot.order = 1 + len(rr[1])
        if newRoot.order != self.ops[rr[0]]:
            raise
        newRoot.offspring.append(self.root)
        self.root.parent = newRoot
        self.root = newRoot
        self.nops[self.root.value] += 1
        self.nodes.append(self.root)
        self.size += 1
        oldRoot = self.root.offspring[0]
        for leaf in rr[1]:
            self.root.offspring.append(Node(leaf, offspring=[], parent=self.root))
            self.nodes.append(self.root.offspring[-1])
            self.ets[0].append(self.root.offspring[-1])
            self.size += 1
        # Add new root to elementary trees if necessary (that is, iff
        # the old root was a leaf)
        if oldRoot.offspring is []:
            self.ets[self.root.order].append(self.root)
        # Update list of distinct parameters
        self.dist_par = list(
            set([n.value for n in self.ets[0] if n.value in self.parameters])
        )
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return self.root

    # -------------------------------------------------------------------------
    def is_root_prunable(self):
        """
        Check if the root is "prunable"

        Returns: boolean of root "prunability"
        """
        if self.size == 1:
            isPrunable = False
        elif self.size == 2:
            isPrunable = True
        else:
            isPrunable = True
            for o in self.root.offspring[1:]:
                if o.offspring != []:
                    isPrunable = False
                    break
        return isPrunable

    # -------------------------------------------------------------------------
    def prune_root(self, update_gof=True, verbose=False):
        """
        Cut the root and its rightmost leaves (provided they are, indeed, leaves),
        leaving the leftmost branch as the new tree. Returns the pruned root with the same format
        as the replacement roots in self.rr_space (or None if pruning was impossible)

        Returns: the replacement root
        """
        # Check if the root is "prunable" (and return None if not)
        if not self.is_root_prunable():
            return None
        # Let's do it!
        rr = [self.root.value, []]
        self.nodes.remove(self.root)
        try:
            self.ets[len(self.root.offspring)].remove(self.root)
        except ValueError:
            pass
        self.nops[self.root.value] -= 1
        self.size -= 1
        for o in self.root.offspring[1:]:
            rr[1].append(o.value)
            self.nodes.remove(o)
            self.size -= 1
            self.ets[0].remove(o)
        self.root = self.root.offspring[0]
        self.root.parent = None
        # Update list of distinct parameters
        self.dist_par = list(
            set([n.value for n in self.ets[0] if n.value in self.parameters])
        )
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        # Done
        return rr

    # -------------------------------------------------------------------------
    def _add_et(self, node, et_order=None, et=None, update_gof=True, verbose=False):
        """
        Add an elementary tree replacing the node, which must be a leaf

        Returns: the input node
        """
        if node.offspring != []:
            raise
        # If no ET is provided, randomly choose one (of the specified
        # order if given, or totally at random otherwise)
        if et is None:
            if et_order is not None:
                et = choice(self.et_space[et_order])
            else:
                all_ets = []
                for o in [o for o in self.op_orders if o > 0]:
                    all_ets += self.et_space[o]
                et = choice(all_ets)
                et_order = len(et[1])
        else:
            et_order = len(et[1])
        # Update the node and its offspring
        node.value = et[0]
        try:
            self.nops[node.value] += 1
        except KeyError:
            pass
        node.offspring = [Node(v, parent=node, offspring=[]) for v in et[1]]
        self.ets[et_order].append(node)
        try:
            self.ets[len(node.parent.offspring)].remove(node.parent)
        except ValueError:
            pass
        except AttributeError:
            pass
        # Add the offspring to the list of nodes
        for n in node.offspring:
            self.nodes.append(n)
        # Remove the node from the list of leaves and add its offspring
        self.ets[0].remove(node)
        for o in node.offspring:
            self.ets[0].append(o)
            self.size += 1
        # Update list of distinct parameters
        self.dist_par = list(
            set([n.value for n in self.ets[0] if n.value in self.parameters])
        )
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def _del_et(self, node, leaf=None, update_gof=True, verbose=False):
        """
        Remove an elementary tree, replacing it by a leaf

        Returns: input node
        """
        if self.size == 1:
            return None
        if leaf is None:
            leaf = choice(self.et_space[0])[0]
        self.nops[node.value] -= 1
        node.value = leaf
        self.ets[len(node.offspring)].remove(node)
        self.ets[0].append(node)
        for o in node.offspring:
            self.ets[0].remove(o)
            self.nodes.remove(o)
            self.size -= 1
        node.offspring = []
        if node.parent is not None:
            is_parent_et = True
            for o in node.parent.offspring:
                if o not in self.ets[0]:
                    is_parent_et = False
                    break
            if is_parent_et:
                self.ets[len(node.parent.offspring)].append(node.parent)
        # Update list of distinct parameters
        self.dist_par = list(
            set([n.value for n in self.ets[0] if n.value in self.parameters])
        )
        self.n_dist_par = len(self.dist_par)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
            self.E = self.get_energy(verbose=verbose)
        return node

    # -------------------------------------------------------------------------
    def et_replace(self, target, new, update_gof=True, verbose=False):
        """
        Replace one elementary tree with another one, both of arbitrary order. target is a
        Node and new is a tuple [node_value, [list, of, offspring, values]]

        Returns: target
        """
        oini, ofin = len(target.offspring), len(new[1])
        if oini == 0:
            added = self._add_et(target, et=new, update_gof=False, verbose=verbose)
        else:
            if ofin == 0:
                added = self._del_et(
                    target, leaf=new[0], update_gof=False, verbose=verbose
                )
            else:
                self._del_et(target, update_gof=False, verbose=verbose)
                added = self._add_et(target, et=new, update_gof=False, verbose=verbose)
        # Update goodness of fit measures, if necessary
        if update_gof:
            self.sse = self.get_sse(verbose=verbose)
            self.bic = self.get_bic(verbose=verbose)
        # Done
        return added

    # -------------------------------------------------------------------------
    def get_sse(self, fit=True, verbose=False):
        """
        Get the sum of squared errors, fitting the expression represented by the Tree
        to the existing data, if specified (by default, yes)

        Returns: sum of square errors (sse)
        """
        # Return 0 if there is no data
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            self.sse = 0
            return 0
        # Convert the Tree into a SymPy expression
        ex = sympify(str(self))
        # Convert the expression to a function that can be used by
        # curve_fit, i.e. that takes as arguments (x, a0, a1, ..., an)
        atomd = dict([(a.name, a) for a in ex.atoms() if a.is_Symbol])
        variables = [atomd[v] for v in self.variables if v in list(atomd.keys())]
        parameters = [atomd[p] for p in self.parameters if p in list(atomd.keys())]
        try:
            flam = lambdify(
                variables + parameters,
                ex,
                [
                    "numpy",
                    {
                        "fac": scipy.special.factorial,
                        "sig": scipy.special.expit,
                        "relu": relu,
                    },
                ],
            )
        except (SyntaxError, KeyError):
            self.sse = dict([(ds, np.inf) for ds in self.x])
            return self.sse
        if fit:
            if len(parameters) == 0:  # Nothing to fit
                for ds in self.x:
                    for p in self.parameters:
                        self.par_values[ds][p] = 1.0
            elif str(self) in self.fit_par:  # Recover previously fit parameters
                self.par_values = self.fit_par[str(self)]
            else:  # Do the fit for all datasets
                self.fit_par[str(self)] = {}
                for ds in self.x:
                    this_x, this_y = self.x[ds], self.y[ds]
                    xmat = [this_x[v.name] for v in variables]

                    def feval(x, *params):
                        args = [xi for xi in x] + [p for p in params]
                        return flam(*args)

                    try:
                        # Fit the parameters
                        res = curve_fit(
                            feval,
                            xmat,
                            this_y,
                            p0=[self.par_values[ds][p.name] for p in parameters],
                            maxfev=10000,
                        )
                        # Reassign the values of the parameters
                        self.par_values[ds] = dict(
                            [
                                (parameters[i].name, res[0][i])
                                for i in range(len(res[0]))
                            ]
                        )
                        for p in self.parameters:
                            if p not in self.par_values[ds]:
                                self.par_values[ds][p] = 1.0
                        # Save this fit
                        self.fit_par[str(self)][ds] = deepcopy(self.par_values[ds])
                    except RuntimeError:
                        # Save this (unsuccessful) fit and print warning
                        self.fit_par[str(self)][ds] = deepcopy(self.par_values[ds])
                        if verbose:
                            print(
                                "#Cannot_fit:%s # # # # #" % str(self).replace(" ", ""),
                                file=sys.stderr,
                            )

        # Sum of squared errors
        self.sse = {}
        for ds in self.x:
            this_x, this_y = self.x[ds], self.y[ds]
            xmat = [this_x[v.name] for v in variables]
            ar = [np.array(xi) for xi in xmat] + [
                self.par_values[ds][p.name] for p in parameters
            ]
            try:
                se = np.square(this_y - flam(*ar))
                if sum(np.isnan(se)) > 0:
                    raise ValueError
                else:
                    self.sse[ds] = np.sum(se)
            except ValueError:
                if verbose:
                    print("> Cannot calculate SSE for %s: inf" % self, file=sys.stderr)
                self.sse[ds] = np.inf

        # Done
        return self.sse

    # -------------------------------------------------------------------------
    def get_bic(self, reset=True, fit=False, verbose=False):
        """
        Calculate the Bayesian information criterion (BIC) of the current expression,
        given the data. If reset==False, the value of self.bic will not be updated
        (by default, it will)

        Returns: Bayesian information criterion (BIC)
        """
        if list(self.x.values())[0].empty or list(self.y.values())[0].empty:
            if reset:
                self.bic = 0
            return 0
        # Get the sum of squared errors (fitting, if required)
        sse = self.get_sse(fit=fit, verbose=verbose)
        # Calculate the BIC
        parameters = set([p.value for p in self.ets[0] if p.value in self.parameters])
        k = 1 + len(parameters)
        BIC = 0.0
        for ds in self.y:
            n = len(self.y[ds])
            BIC += (k - n) * np.log(n) + n * (np.log(2.0 * np.pi) + log(sse[ds]) + 1)
        for ds in self.y:
            if sse[ds] == 0.0:
                BIC = -np.inf
        if reset:
            self.bic = BIC
        return BIC

    # -------------------------------------------------------------------------
    def get_energy(self, bic=False, reset=False, verbose=False):
        """
        Calculate the "energy" of a given formula, that is, approximate minus log-posterior
        of the formula given the data (the approximation coming from the use of the BIC
        instead of the exactly integrated likelihood)

        Returns: Energy of formula (as E, EB, and EP)
        """
        # Contribution of the data (recalculating BIC if necessary)
        if bic:
            EB = self.get_bic(reset=reset, verbose=verbose) / 2.0
        else:
            EB = self.bic / 2.0
        # Contribution from the prior
        EP = 0.0
        for op, nop in list(self.nops.items()):
            try:
                EP += self.prior_par["Nopi_%s" % op] * nop
            except KeyError:
                pass
            try:
                EP += self.prior_par["Nopi2_%s" % op] * nop**2
            except KeyError:
                pass
        # Reset the value, if necessary
        if reset:
            self.EB = EB
            self.EP = EP
            self.E = EB + EP
        # Done
        return EB + EP, EB, EP

    # -------------------------------------------------------------------------
    def update_representative(self, verbose=False):
        """Check if we've seen this formula before, either in its current form
        or in another form.

        *If we haven't seen it, save it and return 1.

        *If we have seen it and this IS the representative, just return 0.

        *If we have seen it and the representative has smaller energy, just return -1.

        *If we have seen it and the representative has higher energy, update
        the representatitve and return -2.

        Returns: Integer value (0, 1, or -1) corresponding to:
            0: we have seen this canonical form before
            1: we haven't seen this canonical form before
            -1: we have seen this equation's canonical form before but it isn't in that form yet
        """
        # Check for canonical representative
        canonical = self.canonical(verbose=verbose)
        try:  # We've seen this canonical before!
            rep, rep_energy, rep_par_values = self.representative[canonical]
        except KeyError:  # Never seen this canonical formula before:
            # save it and return 1
            self.get_bic(reset=True, fit=True, verbose=verbose)
            new_energy = self.get_energy(bic=False, verbose=verbose)
            self.representative[canonical] = (
                str(self),
                new_energy,
                deepcopy(self.par_values),
            )
            return 1

        # If we've seen this canonical before, check if the
        # representative needs to be updated
        if rep == str(self):  # This IS the representative: return 0
            return 0
        else:
            return -1

    # -------------------------------------------------------------------------
    def dE_et(self, target, new, verbose=False):
        """
        Calculate the energy change associated to the replacement of one elementary tree
        with another, both of arbitrary order. "target" is a Node() and "new" is
        a tuple [node_value, [list, of, offspring, values]].

        Returns: change in energy associated with an elementary tree replacement move
        """
        dEB, dEP = 0.0, 0.0

        # Some terms of the acceptance (number of possible move types
        # from initial and final configurations), as well as checking
        # if the tree is canonically acceptable.

        # number of possible move types from initial
        nif = sum(
            [
                int(len(self.ets[oi]) > 0 and (self.size + of - oi) <= self.max_size)
                for oi, of in self.move_types
            ]
        )
        # replace
        old = [target.value, [o.value for o in target.offspring]]
        old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
        old_par_values = deepcopy(self.par_values)
        added = self.et_replace(target, new, update_gof=False, verbose=verbose)
        # number of possible move types from final
        nfi = sum(
            [
                int(len(self.ets[oi]) > 0 and (self.size + of - oi) <= self.max_size)
                for oi, of in self.move_types
            ]
        )
        # check/update canonical representative
        rep_res = self.update_representative(verbose=verbose)
        if rep_res == -1:
            # this formula is forbidden
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values
            return np.inf, np.inf, np.inf, deepcopy(self.par_values), nif, nfi
        # leave the whole thing as it was before the back & fore
        self.et_replace(added, old, update_gof=False, verbose=verbose)
        self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
        self.par_values = old_par_values
        # Prior: change due to the numbers of each operation
        try:
            dEP -= self.prior_par["Nopi_%s" % target.value]
        except KeyError:
            pass
        try:
            dEP += self.prior_par["Nopi_%s" % new[0]]
        except KeyError:
            pass
        try:
            dEP += self.prior_par["Nopi2_%s" % target.value] * (
                (self.nops[target.value] - 1) ** 2 - (self.nops[target.value]) ** 2
            )
        except KeyError:
            pass
        try:
            dEP += self.prior_par["Nopi2_%s" % new[0]] * (
                (self.nops[new[0]] + 1) ** 2 - (self.nops[new[0]]) ** 2
            )
        except KeyError:
            pass

        # Data
        if not list(self.x.values())[0].empty:
            bicOld = self.bic
            sseOld = deepcopy(self.sse)
            par_valuesOld = deepcopy(self.par_values)
            old = [target.value, [o.value for o in target.offspring]]
            # replace
            added = self.et_replace(target, new, update_gof=True, verbose=verbose)
            bicNew = self.bic
            par_valuesNew = deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.et_replace(added, old, update_gof=False, verbose=verbose)
            self.bic = bicOld
            self.sse = deepcopy(sseOld)
            self.par_values = par_valuesOld
            dEB += (bicNew - bicOld) / 2.0
        else:
            par_valuesNew = deepcopy(self.par_values)
        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
        except (ValueError, TypeError):
            dEB, dEP, dE = np.inf, np.inf, np.inf
        return dE, dEB, dEP, par_valuesNew, nif, nfi

    # -------------------------------------------------------------------------
    def dE_lr(self, target, new, verbose=False):
        """
        Calculate the energy change associated to a long-range move
        (the replacement of the value of a node. "target" is a Node() and "new" is a node_value

        Returns: energy change associated with a long-range move
        """
        dEB, dEP = 0.0, 0.0
        par_valuesNew = deepcopy(self.par_values)

        if target.value != new:

            # Check if the new tree is canonically acceptable.
            old = target.value
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            target.value = new
            try:
                self.nops[old] -= 1
                self.nops[new] += 1
            except KeyError:
                pass
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                target.value = old
                try:
                    self.nops[old] += 1
                    self.nops[new] -= 1
                except KeyError:
                    pass
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, None
            # leave the whole thing as it was before the back & fore
            target.value = old
            try:
                self.nops[old] += 1
                self.nops[new] -= 1
            except KeyError:
                pass
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            try:
                dEP -= self.prior_par["Nopi_%s" % target.value]
            except KeyError:
                pass
            try:
                dEP += self.prior_par["Nopi_%s" % new]
            except KeyError:
                pass
            try:
                dEP += self.prior_par["Nopi2_%s" % target.value] * (
                    (self.nops[target.value] - 1) ** 2 - (self.nops[target.value]) ** 2
                )
            except KeyError:
                pass
            try:
                dEP += self.prior_par["Nopi2_%s" % new] * (
                    (self.nops[new] + 1) ** 2 - (self.nops[new]) ** 2
                )
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                old = target.value
                target.value = new
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                target.value = old
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)

        # Done
        try:
            dEB = float(dEB)
            dEP = float(dEP)
            dE = dEB + dEP
            return dE, dEB, dEP, par_valuesNew
        except (ValueError, TypeError):
            return np.inf, np.inf, np.inf, None

    # -------------------------------------------------------------------------
    def dE_rr(self, rr=None, verbose=False):
        """
        Calculate the energy change associated to a root replacement move.
        If rr==None, then it returns the energy change associated to pruning the root; otherwise,
        it returns the energy change associated to adding the root replacement "rr"

        Returns: energy change associated with a root replacement move
        """
        dEB, dEP = 0.0, 0.0

        # Root pruning
        if rr is None:
            if not self.is_root_prunable():
                return np.inf, np.inf, np.inf, self.par_values

            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            oldrr = [self.root.value, [o.value for o in self.root.offspring[1:]]]
            self.prune_root(update_gof=False, verbose=verbose)
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP -= self.prior_par["Nopi_%s" % self.root.value]
            try:
                dEP += self.prior_par["Nopi2_%s" % self.root.value] * (
                    (self.nops[self.root.value] - 1) ** 2
                    - (self.nops[self.root.value]) ** 2
                )
            except KeyError:
                pass

            # Data correction
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                oldrr = [self.root.value, [o.value for o in self.root.offspring[1:]]]
                # replace
                self.prune_root(update_gof=False, verbose=verbose)
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.replace_root(rr=oldrr, update_gof=False, verbose=verbose)
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except (ValueError, TypeError):
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew

        # Root replacement
        else:
            # Check if the new tree is canonically acceptable.
            # replace
            old_bic, old_sse, old_energy = self.bic, deepcopy(self.sse), self.E
            old_par_values = deepcopy(self.par_values)
            newroot = self.replace_root(rr=rr, update_gof=False, verbose=verbose)
            if newroot is None:  # Root cannot be replaced (due to max_size)
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # check/update canonical representative
            rep_res = self.update_representative(verbose=verbose)
            if rep_res == -1:
                # this formula is forbidden
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
                self.par_values = old_par_values
                return np.inf, np.inf, np.inf, deepcopy(self.par_values)
            # leave the whole thing as it was before the back & fore
            self.prune_root(update_gof=False, verbose=verbose)
            self.bic, self.sse, self.E = old_bic, deepcopy(old_sse), old_energy
            self.par_values = old_par_values

            # Prior: change due to the numbers of each operation
            dEP += self.prior_par["Nopi_%s" % rr[0]]
            try:
                dEP += self.prior_par["Nopi2_%s" % rr[0]] * (
                    (self.nops[rr[0]] + 1) ** 2 - (self.nops[rr[0]]) ** 2
                )
            except KeyError:
                pass

            # Data
            if not list(self.x.values())[0].empty:
                bicOld = self.bic
                sseOld = deepcopy(self.sse)
                par_valuesOld = deepcopy(self.par_values)
                # replace
                newroot = self.replace_root(rr=rr, update_gof=False, verbose=verbose)
                if newroot is None:
                    return np.inf, np.inf, np.inf, self.par_values
                bicNew = self.get_bic(reset=True, fit=True, verbose=verbose)
                par_valuesNew = deepcopy(self.par_values)
                # leave the whole thing as it was before the back & fore
                self.prune_root(update_gof=False, verbose=verbose)
                self.bic = bicOld
                self.sse = deepcopy(sseOld)
                self.par_values = par_valuesOld
                dEB += (bicNew - bicOld) / 2.0
            else:
                par_valuesNew = deepcopy(self.par_values)
            # Done
            try:
                dEB = float(dEB)
                dEP = float(dEP)
                dE = dEB + dEP
            except (ValueError, TypeError):
                dEB, dEP, dE = np.inf, np.inf, np.inf
            return dE, dEB, dEP, par_valuesNew

    # -------------------------------------------------------------------------
    def mcmc_step(self, verbose=False, p_rr=0.05, p_long=0.45):
        """
        Make a single MCMC step

        Returns: None or expression list
        """
        topDice = random()
        # Root replacement move
        if topDice < p_rr:
            if random() < 0.5:
                # Try to prune the root
                dE, dEB, dEP, par_valuesNew = self.dE_rr(rr=None, verbose=verbose)
                if -dEB / self.BT - dEP / self.PT > 300:
                    paccept = 1
                else:
                    paccept = np.exp(-dEB / self.BT - dEP / self.PT) / float(
                        self.num_rr
                    )
                dice = random()
                if dice < paccept:
                    # Accept move
                    self.prune_root(update_gof=False, verbose=verbose)
                    self.par_values = par_valuesNew
                    self.get_bic(reset=True, fit=False, verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP
            else:
                # Try to replace the root
                newrr = choice(self.rr_space)
                dE, dEB, dEP, par_valuesNew = self.dE_rr(rr=newrr, verbose=verbose)
                if self.num_rr > 0 and -dEB / self.BT - dEP / self.PT > 0:
                    paccept = 1.0
                elif self.num_rr == 0:
                    paccept = 0.0
                else:
                    paccept = self.num_rr * np.exp(-dEB / self.BT - dEP / self.PT)
                dice = random()
                if dice < paccept:
                    # Accept move
                    self.replace_root(rr=newrr, update_gof=False, verbose=verbose)
                    self.par_values = par_valuesNew
                    self.get_bic(reset=True, fit=False, verbose=verbose)
                    self.E += dE
                    self.EB += dEB
                    self.EP += dEP

        # Long-range move
        elif topDice < (p_rr + p_long):
            # Choose a random node in the tree, and a random new operation
            target = choice(self.nodes)
            nready = False
            while not nready:
                if len(target.offspring) == 0:
                    new = choice(self.variables + self.parameters)
                    nready = True
                else:
                    new = choice(list(self.ops.keys()))
                    if self.ops[new] == self.ops[target.value]:
                        nready = True
            dE, dEB, dEP, par_valuesNew = self.dE_lr(target, new, verbose=verbose)
            try:
                paccept = np.exp(-dEB / self.BT - dEP / self.PT)
            except ValueError:
                _logger.warning("Potentially failing to set paccept properly")
                if (dEB / self.BT + dEP / self.PT) < 0:
                    paccept = 1.0
            # Accept move, if necessary
            dice = random()
            if dice < paccept:
                # update number of operations
                if target.offspring != []:
                    self.nops[target.value] -= 1
                    self.nops[new] += 1
                # move
                target.value = new
                # recalculate distinct parameters
                self.dist_par = list(
                    set([n.value for n in self.ets[0] if n.value in self.parameters])
                )
                self.n_dist_par = len(self.dist_par)
                # update others
                self.par_values = deepcopy(par_valuesNew)
                self.get_bic(reset=True, fit=False, verbose=verbose)
                self.E += dE
                self.EB += dEB
                self.EP += dEP

        # Elementary tree (short-range) move
        else:
            # Choose a feasible move (doable and keeping size<=max_size)
            while True:
                oini, ofin = choice(self.move_types)
                if len(self.ets[oini]) > 0 and (
                    self.size - oini + ofin <= self.max_size
                ):
                    break
            # target and new ETs
            target = choice(self.ets[oini])
            new = choice(self.et_space[ofin])
            # omegai and omegaf
            omegai = len(self.ets[oini])
            omegaf = len(self.ets[ofin]) + 1
            if ofin == 0:
                omegaf -= oini
            if oini == 0 and target.parent in self.ets[ofin]:
                omegaf -= 1
            # size of et_space of each type
            si = len(self.et_space[oini])
            sf = len(self.et_space[ofin])
            # Probability of acceptance
            dE, dEB, dEP, par_valuesNew, nif, nfi = self.dE_et(
                target, new, verbose=verbose
            )
            try:
                paccept = (
                    float(nif) * omegai * sf * np.exp(-dEB / self.BT - dEP / self.PT)
                ) / (float(nfi) * omegaf * si)
            except ValueError:
                if (dEB / self.BT + dEP / self.PT) < -200:
                    paccept = 1.0
            # Accept / reject
            dice = random()
            if dice < paccept:
                # Accept move
                self.et_replace(target, new, verbose=verbose)
                self.par_values = par_valuesNew
                self.get_bic(verbose=verbose)
                self.E += dE
                self.EB += dEB
                self.EP += dEP

        # Done
        return

    # -------------------------------------------------------------------------
    def mcmc(
        self,
        tracefn="trace.dat",
        progressfn="progress.dat",
        write_files=True,
        reset_files=True,
        burnin=2000,
        thin=10,
        samples=10000,
        verbose=False,
        progress=True,
    ):
        """
        Sample the space of formula trees using MCMC, and write the trace and some progress
        information to files (unless write_files is False)

        Returns: None or expression list
        """
        self.get_energy(reset=True, verbose=verbose)

        # Burning
        if progress:
            sys.stdout.write("# Burning in\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))
        for i in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and (i % (burnin / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
        # Sample
        if write_files:
            if reset_files:
                tracef = open(tracefn, "w")
                progressf = open(progressfn, "w")
            else:
                tracef = open(tracefn, "a")
                progressf = open(progressfn, "a")
        if progress:
            sys.stdout.write("\n# Sampling\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))
        for s in range(samples):
            for i in range(thin):
                self.mcmc_step(verbose=verbose)
            if progress and (s % (samples / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
            if write_files:
                json.dump(
                    [
                        s,
                        float(self.bic),
                        float(self.E),
                        str(self.get_energy(verbose=verbose)),
                        str(self),
                        self.par_values,
                    ],
                    tracef,
                )
                tracef.write("\n")
                tracef.flush()
                progressf.write("%d %lf %lf\n" % (s, self.E, self.bic))
                progressf.flush()
        # Done
        if progress:
            sys.stdout.write("\n")
        return

    # -------------------------------------------------------------------------
    def predict(self, x):
        """
        Calculate the value of the formula at the given data x. The data x
        must have the same format as the training data and, in particular, it
        it must specify to which dataset the example data belongs, if multiple
        datasets where used for training.

        Returns: predicted y values
        """
        if isinstance(x, pd.DataFrame):
            this_x = {"d0": x}
            input_type = "df"
        elif isinstance(x, dict):
            this_x = x
            input_type = "dict"
        else:
            raise TypeError("x must be either a dict or a pandas.DataFrame")

        # Convert the Tree into a SymPy expression
        ex = sympify(str(self))
        # Convert the expression to a function
        atomd = dict([(a.name, a) for a in ex.atoms() if a.is_Symbol])
        variables = [atomd[v] for v in self.variables if v in list(atomd.keys())]
        parameters = [atomd[p] for p in self.parameters if p in list(atomd.keys())]
        flam = lambdify(
            variables + parameters,
            ex,
            [
                "numpy",
                {
                    "fac": scipy.special.factorial,
                    "sig": scipy.special.expit,
                    "relu": relu,
                },
            ],
        )
        # Loop over datasets
        predictions = {}
        for ds in this_x:
            # Prepare variables and parameters
            xmat = [this_x[ds][v.name] for v in variables]
            params = [self.par_values[ds][p.name] for p in parameters]
            args = [xi for xi in xmat] + [p for p in params]
            # Predict
            try:
                prediction = flam(*args)
            except SyntaxError:
                # Do it point by point
                prediction = [np.nan for i in range(len(this_x[ds]))]
            predictions[ds] = pd.Series(prediction, index=list(this_x[ds].index))

        if input_type == "df":
            return predictions["d0"]
        else:
            return predictions

    # -------------------------------------------------------------------------
    def trace_predict(
        self,
        x,
        burnin=1000,
        thin=2000,
        samples=1000,
        tracefn="trace.dat",
        progressfn="progress.dat",
        write_files=False,
        reset_files=True,
        verbose=False,
        progress=True,
    ):
        """
        Sample the space of formula trees using MCMC,
        and predict y(x) for each of the sampled formula trees

        Returns: predicted y values for each of the sampled formula trees
        """
        ypred = {}
        # Burning
        if progress:
            sys.stdout.write("# Burning in\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))
        for i in range(burnin):
            self.mcmc_step(verbose=verbose)
            if progress and (i % (burnin / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
        # Sample
        if write_files:
            if reset_files:
                tracef = open(tracefn, "w")
                progressf = open(progressfn, "w")
            else:
                tracef = open(tracefn, "a")
                progressf = open(progressfn, "a")
        if progress:
            sys.stdout.write("\n# Sampling\t")
            sys.stdout.write("[%s]" % (" " * 50))
            sys.stdout.flush()
            sys.stdout.write("\b" * (50 + 1))

        for s in range(samples):
            for kk in range(thin):
                self.mcmc_step(verbose=verbose)
            # Make prediction
            ypred[s] = self.predict(x)
            # Output
            if progress and (s % (samples / 50) == 0):
                sys.stdout.write("=")
                sys.stdout.flush()
            if write_files:
                json.dump(
                    [
                        s,
                        float(self.bic),
                        float(self.E),
                        float(self.get_energy(verbose=verbose)),
                        str(self),
                        self.par_values,
                    ],
                    tracef,
                )
                tracef.write("\n")
                tracef.flush()
                progressf.write("%d %lf %lf\n" % (s, self.E, self.bic))
                progressf.flush()
        # Done
        if progress:
            sys.stdout.write("\n")
        return pd.DataFrame.from_dict(ypred)


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------


def test3(num_points=10, samples=100000):
    # Create the data
    x = pd.DataFrame(
        dict([("x%d" % i, np.random.uniform(0, 10, num_points)) for i in range(5)])
    )
    eps = np.random.normal(0.0, 5, num_points)
    y = 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3 + eps
    x.to_csv("data_x.csv", index=False)
    y.to_csv("data_y.csv", index=False, header=["y"])

    # Create the formula
    prior_par, _ = get_priors()
    t = Tree(
        variables=["x%d" % i for i in range(5)],
        parameters=["a%d" % i for i in range(10)],
        x=x,
        y=y,
        prior_par=prior_par,
        BT=1.0,
    )
    # MCMC
    t.mcmc(burnin=2000, thin=10, samples=samples, verbose=True)

    # Predict
    print(t.predict(x))
    print(y)
    print(50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)

    plt.plot(t.predict(x), 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3)
    plt.show()

    return t


def test4(num_points=10, samples=1000):
    # Create the data
    x = pd.DataFrame(
        dict([("x%d" % i, np.random.uniform(0, 10, num_points)) for i in range(5)])
    )
    eps = np.random.normal(0.0, 5, num_points)
    y = 50.0 * np.sin(x["x0"]) / x["x2"] - 4.0 * x["x1"] + 3 + eps
    x.to_csv("data_x.csv", index=False)
    y.to_csv("data_y.csv", index=False, header=["y"])

    xtrain, ytrain = x.iloc[5:], y.iloc[5:]
    xtest, ytest = x.iloc[:5], y.iloc[:5]

    # Create the formula
    prior_par, _ = get_priors()
    t = Tree(
        variables=["x%d" % i for i in range(5)],
        parameters=["a%d" % i for i in range(10)],
        x=xtrain,
        y=ytrain,
        prior_par=prior_par,
    )
    print(xtest)

    # Predict
    ypred = t.trace_predict(xtest, samples=samples, burnin=10000)

    print(ypred)
    print(ytest)
    print(50.0 * np.sin(xtest["x0"]) / xtest["x2"] - 4.0 * xtest["x1"] + 3)

    # Done
    return t


def test5(string="(P120 + (((ALPHACAT / _a2) + (_a2 * CDH3)) + _a0))"):
    # Create the formula
    prior_par, _ = get_priors("GuimeraTest2020")

    t = Tree(prior_par=prior_par, from_string=string)
    for i in range(1000000):
        t.mcmc_step(verbose=True)
        print("-" * 150)
        t2 = Tree(from_string=str(t))
        print(t)
        print(t2)
        if str(t2) != str(t):
            raise

    return t


if __name__ == "__main__":
    NP, NS = 100, 1000
    test5()
