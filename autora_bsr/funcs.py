import numpy as np
from scipy.stats import invgamma
from scipy.stats import norm
import copy


class Operator:
    def __init__(self, name, function, arity):
        self.name = name
        self.func = function
        self.arity = arity  # num of inputs


class Node:
    def __init__(self, depth):
        # tree structure attributes
        self.type = -1
        # -1 represents newly grown node (not decided yet)
        # 0 represents no child, as a terminal node
        # 1 represents one child,
        # 2 represents 2 children
        self.order = 0
        self.left = None
        self.right = None
        # if type=1, the left child is the only one
        self.depth = depth
        self.parent = None

        # calculation attributes
        self.operator = None
        self.op_ind = None
        # operator is a string, either "+","*","ln","exp","inv"
        self.data = None
        self.feature = None
        # feature is a int indicating the index of feature in the input data
        # possible parameters
        self.a = None
        self.b = None

    def inform(self):
        print("order:", self.order)
        print("type:", self.type)
        print("depth:", self.depth)
        print("operator:", self.operator)
        print("data:", self.data)
        print("feature:", self.feature)
        if self.operator == 'ln':
            print(" ln_a:", self.a)
            print(" ln_b:", self.b)

        return


# =============================================================================
# # grow from a node, assign an operator or stop as terminal
# =============================================================================

def grow(node, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b):
    depth = node.depth

    # deciding the number of child nodes
    if node.depth > 0:
        prob = 1 / np.power((1 + depth), -beta)

        test = np.random.uniform(0, 1, 1)
        if test > prob:  # terminal
            node.feature = np.random.randint(0, nfeature, size=1)
            node.type = 0
        else:
            op_ind = np.random.choice(np.arange(len(Ops)), p=Op_weights)
            node.operator = Ops[op_ind]
            node.type = Op_type[op_ind]
            node.op_ind = op_ind

    else:  # root node, sure to split
        op_ind = np.random.choice(np.arange(len(Ops)), p=Op_weights)
        node.operator = Ops[op_ind]
        node.type = Op_type[op_ind]
        node.op_ind = op_ind

    # grow recursively
    if node.type == 0:
        node.feature = np.random.randint(0, nfeature, size=1)

    elif node.type == 1:
        node.left = Node(depth + 1)
        node.left.parent = node
        if node.operator == 'ln':  # linear parameters
            node.a = norm.rvs(loc=1, scale=np.sqrt(sigma_a))
            node.b = norm.rvs(loc=0, scale=np.sqrt(sigma_b))
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

    else:  # node.type=2
        node.left = Node(depth + 1)
        node.left.parent = node
        # node.left.order = len(Tree)
        node.right = Node(depth + 1)
        node.right.parent = node
        # node.right.order = len(Tree)
        grow(node.left, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        grow(node.right, nfeature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

    return


# =============================================================================
# # generate a list storing the nodes in the tree
# # nodes are stored by induction
# # orders are assigned accordingly
# =============================================================================
def genList(node):
    lst = []
    # terminal node
    if node.left is None:
        lst.append(node)
    else:
        if node.right is None:  # with one child
            lst.append(node)
            lst = lst + genList(node.left)
        else:  # with two children
            lst.append(node)
            lst = lst + genList(node.left)
            lst = lst + genList(node.right)
    for i in np.arange(0, len(lst)):
        lst[i].order = i
    return (lst)


# =============================================================================
# # cut all child nodes of a current node
# # turn the node into a terminal one
# =============================================================================
def shrink(node):
    if node.left is None:
        print("Already a terminal node!")
    else:
        node.left = None
        node.right = None
        node.type = 0
        node.operator = None  # delete operator
        node.a = None  # delete parameters
        node.b = None
    return


# =============================================================================
# # upgrade 'order' attribute of nodes in Tree
# # Tree is a list containing nodes of a tree
# =============================================================================
def upgOd(Tree):
    for i in np.arange(0, len(Tree)):
        Tree[i].order = i
    return


# =============================================================================
# # calculate a tree output from node
# =============================================================================
def allcal(node, indata):
    if node.type == 0:  # terminal node
        if indata is not None:
            node.data = np.array(indata.iloc[:, node.feature])
    elif node.type == 1:  # one child node
        if node.operator == 'ln':
            node.data = node.a * allcal(node.left, indata) + node.b
        elif node.operator == 'exp':
            node.data = allcal(node.left, indata)
            for i in np.arange(len(node.data[:, 0])):
                if node.data[i, 0] <= 200:
                    node.data[i, 0] = np.exp(node.data[i, 0])
                else:
                    node.data[i, 0] = 1e+10
        elif node.operator == 'inv':
            node.data = allcal(node.left, indata)
            for i in np.arange(len(node.data[:, 0])):
                if node.data[i, 0] == 0:
                    node.data[i, 0] = 0
                else:
                    node.data[i, 0] = 1 / node.data[i, 0]
        elif node.operator == 'neg':
            node.data = -1 * allcal(node.left, indata)
        elif node.operator == 'sin':
            node.data = np.sin(allcal(node.left, indata))
        elif node.operator == 'cos':
            node.data = np.cos(allcal(node.left, indata))
        elif node.operator == 'square': ## operator added by fwl
            node.data = np.square(allcal(node.left, indata))
        elif node.operator == 'cubic': ## operator added by fwl
            node.data = np.power(allcal(node.left, indata),3)
        else:
            print("No matching type and operator!")
    elif node.type == 2:  # two child nodes
        if node.operator == '+':
            node.data = allcal(node.left, indata) + allcal(node.right, indata)
        elif node.operator == '*':
            node.data = allcal(node.left, indata) * allcal(node.right, indata)
        else:
            print("No matching type and operator!")
    elif node.type == -1:  # not grown
        print("Not a grown tree!")
    else:
        print("No legal node type!")

    return node.data


# =============================================================================
# # display the structure of the tree, each node displays operator
# # Tree is a list storing the nodes
# =============================================================================
def display(Tree):
    tree_depth = -1
    for i in np.arange(0, len(Tree)):
        if Tree[i].depth > tree_depth:
            tree_depth = Tree[i].depth
    dlists = []
    for d in np.arange(0, tree_depth + 1):
        dlists.append([])
    for i in np.arange(0, len(Tree)):
        dlists[Tree[i].depth].append(Tree[i])

    for d in np.arange(0, len(dlists)):
        st = " "
        for i in np.arange(0, len(dlists[d])):
            if dlists[d][i].type > 0:
                st = st + dlists[d][i].operator + " "
            else:
                st = st + str(dlists[d][i].feature) + " "
        print(st)
    return


# =============================================================================
# # get the height of a (sub)tree with node being root node
# # equivalently, the maximum distance from node to its descendent
# # only a root node has height 0
# # terminal nodes has height 0
# =============================================================================
def getHeight(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        return getHeight(node.left) + 1
    else:
        lheight = getHeight(node.left)
        rheight = getHeight(node.right)
        return max(lheight, rheight) + 1


# =============================================================================
# # get the number of nodes of a (sub)tree with node being root
# =============================================================================
def getNum(node):
    if node.type == 0:
        return 1
    elif node.type == 1:
        return getNum(node.left) + 1
    else:
        lnum = getNum(node.left)
        rnum = getNum(node.right)
        return (lnum + rnum + 1)


# =============================================================================
# # get the number of lt() operators of a (sub)tree with node being root
# =============================================================================
def numLT(node):
    if node.type == 0:
        return 0
    elif node.type == 1:
        if node.operator == 'ln':
            return 1 + numLT(node.left)
        else:
            return numLT(node.left)
    else:
        return numLT(node.left) + numLT(node.right)


# =============================================================================
# # update depth of all nodes
# =============================================================================
def upDepth(Root):
    if Root.parent is None:
        Root.depth = 0
    else:
        Root.depth = Root.parent.depth + 1

    if Root.left is not None:
        upDepth(Root.left)
        if Root.right is not None:
            upDepth(Root.right)


# =============================================================================
# # returns a string of the expression of the tree
# # node is the root of the tree
# =============================================================================
def Express(node):
    expr = ""
    if node.type == 0:  # terminal
        expr = "x" + str(node.feature)
        return (expr)
    elif node.type == 1:
        if node.operator == 'exp':
            expr = "exp(" + Express(node.left) + ")"
        elif node.operator == 'ln':
            expr = str(round(node.a, 4)) + "*(" + Express(node.left) + ")+" + str(round(node.b, 4))
        elif node.operator == 'inv':  # node.operator == 'inv':
            expr = "1/[" + Express(node.left) + "]"
        elif node.operator == 'sin':
            expr = "sin(" + Express(node.left) + ")"
        elif node.operator == 'cos':
            expr = "cos(" + Express(node.left) + ")"
        elif node.operator == 'square': ## added by fwl
            expr = "(" + Express(node.left) + ")^2"
        elif node.operator == 'cubic': ## added by fwl
            expr = "(" + Express(node.left) + ")^3"
        else:  # note.operate=='neg'
            expr = "-(" + Express(node.left) + ")"

    else:  # node.type==2
        if node.operator == '+':
            expr = Express(node.left) + "+" + Express(node.right)
        else:
            expr = "(" + Express(node.left) + ")*(" + Express(node.right) + ")"
    return (expr)


# =============================================================================
# # compute the likelihood of tree structure f(S)
# # P(M,T)*P(theta|M,T)*P(theta|sigma_theta)*P(sigma_theta)*P(theta)
# =============================================================================
def fStruc(node, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b):
    loglike = 0  # log.likelihood of structure S=(T,M)
    loglike_para = 0  # log.likelihood of linear paras

    '''
    # contribution of hyperparameter sigma_theta
    if node.depth == 0:#root node
        loglike += np.log(invgamma.pdf(node.sigma_a,1))
        loglike += np.log(invgamma.pdf(node.sigma_b,1))
    '''

    # contribution of splitting the node or becoming terminal
    if node.type == 0:  # terminal node
        loglike += np.log(
            1 - 1 / np.power((1 + node.depth), -beta))  # * np.power(node.depth,beta) #contribution of choosing terminal
        loglike -= np.log(n_feature)  # contribution of feature selection
    elif node.type == 1:  # unitary operator
        # contribution of splitting
        if node.depth == 0:  # root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log((1 + node.depth)) * beta + np.log(Op_weights[node.op_ind])
        # contribution of parameters of linear nodes
        if node.operator == 'ln':
            loglike_para -= np.power((node.a - 1), 2) / (2 * sigma_a)
            loglike_para -= np.power(node.b, 2) / (2 * sigma_b)
            loglike_para -= 0.5 * np.log(2 * np.pi * sigma_a)
            loglike_para -= 0.5 * np.log(2 * np.pi * sigma_b)
    else:  # binary operator
        # contribution of splitting
        if node.depth == 0:  # root node
            loglike += np.log(Op_weights[node.op_ind])
        else:
            loglike += np.log((1 + node.depth)) * beta + np.log(Op_weights[node.op_ind])

    # contribution of child nodes
    if node.left is None:  # no child nodes
        return [loglike, loglike_para]
    else:
        fleft = fStruc(node.left, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        loglike += fleft[0]
        loglike_para += fleft[1]
        if node.right is None:  # only one child
            return [loglike, loglike_para]
        else:
            fright = fStruc(node.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
            loglike += fright[0]
            loglike_para += fright[1]

    return [loglike, loglike_para]


# =============================================================================
# # propose a new tree from existing Root
# # and calculate the ratio
# # five possible actions: stay, grow, prune, ReassignOp, ReassignFea.
# =============================================================================
def Prop(Root, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b):
    ###############################
    ######### preparations ########
    ###############################

    # make a copy of Root
    oldRoot = copy.deepcopy(Root)
    # get necessary auxiliary information
    depth = -1
    Tree = genList(Root)
    for i in np.arange(0, len(Tree)):
        if Tree[i].depth > depth:
            depth = Tree[i].depth

    # preserve pointers to originally linear nodes
    lnPointers = []
    last_a = []
    last_b = []
    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            lnPointers.append(Tree[i])
            last_a.append(Tree[i].a)
            last_b.append(Tree[i].b)

    # get the list of terminal nodes
    Term = []  # terminal
    Nterm = []  # non-terminal
    cnode = None
    for i in np.arange(0, len(Tree)):
        if Tree[i].type == 0:
            Term.append(Tree[i])
        else:
            Nterm.append(Tree[i])

    # get the list of lt() nodes
    Lins = []
    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            Lins.append(Tree[i])
    ltNum = len(Lins)

    # record expansion and shrinkage
    # expansion occurs when num of lt() increases
    # shrinkage occurs when num of lt() decreases
    change = ''
    Q = Qinv = 1

    # qualified candidates for detransformation
    detcd = []
    # for detransformation: not root or root but child nodes are not all terminal
    # transformation can be applied to any node
    for i in np.arange(len(Tree)):
        flag = True
        if Tree[i].type == 0:  # terminal is not allowed
            flag = False
        if Tree[i].parent is None:  # root
            if Tree[i].right is None and Tree[i].left.type == 0:
                flag = False
            elif Tree[i].left.type == 0 and Tree[i].right.type == 0:
                flag = False

        if flag == True:
            detcd.append(Tree[i])

    ###############################
    # decide which action to take #
    ###############################

    # probs of each action
    p_stay = 0.25 * ltNum / (ltNum + 3)
    p_grow = (1 - p_stay) * min(1, 4 / (len(Nterm) + 2)) / 3
    p_prune = (1 - p_stay) / 3 - p_grow
    p_detr = (1 - p_stay) * (1 / 3) * len(detcd) / (3 + len(detcd))
    p_trans = (1 - p_stay) / 3 - p_detr
    p_rop = (1 - p_stay) / 6

    # auxiliary
    test = np.random.uniform(0, 1, 1)[0]

    ###############################
    ########### take action #######
    ###############################

    # stay
    if test <= p_stay:
        action = 'stay'
        # print('action:',action)
        # calculate Q and Qinv
        Q = p_stay
        Qinv = p_stay
        # update all linear nodes 
        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                Tree[i].a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
                Tree[i].b = norm.rvs(loc=1,scale=np.sqrt(sigma_b))

    # grow
    elif test <= p_stay + p_grow:
        action = 'grow'
        # print("action:",action)

        # pick a terminal node
        pod = np.random.randint(0, len(Term), 1)[0]
        # grow the node
        # the likelihood is exactly the same as fStruc(), starting from assigning operator
        grow(Term[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

        if Term[pod].type == 0:  # grow to be terminal
            Q = Qinv = 1
        else:
            # calculate Q
            fstrc = fStruc(Term[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
            Q = p_grow * np.exp(fstrc[0]) / len(Term)
            # calculate Qinv (equiv to prune)
            new_ltNum = numLT(Root)
            new_height = getHeight(Root)
            new_nodeNum = getNum(Root)
            newTerm = []  # terminal
            newTree = genList(Root)
            new_nterm = []  # non-terminal
            for i in np.arange(0, len(newTree)):
                if newTree[i].type == 0:
                    newTerm.append(newTree[i])
                else:
                    new_nterm.append(newTree[i])
            new_termNum = len(newTerm)
            new_p = (1 - 0.25 * new_ltNum / (new_ltNum + 3)) * (1 - min(1, 4 / (len(new_nterm) + 2))) / 3
            Qinv = new_p / max(1, (new_nodeNum - new_termNum - 1))  # except root node

            if new_ltNum > ltNum:
                change = 'expansion'

    # prune
    elif test <= p_stay + p_grow + p_prune:
        action = 'prune'
        # print("action:",action)

        # pick a node to prune
        pod = np.random.randint(1, len(Nterm), 1)[0]  # except root node
        fstrc = fStruc(Nterm[pod], n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        pruned = copy.deepcopy(Nterm[pod])  # preserve a copy

        # preserve pointers to all cutted ln
        p_ltNum = numLT(pruned)
        if p_ltNum > 0:
            change = 'shrinkage'

        # prune the node
        Nterm[pod].left = None
        Nterm[pod].right = None
        Nterm[pod].operator = None
        Nterm[pod].type = 0
        Nterm[pod].feature = np.random.randint(0, n_feature, 1)
        # print("prune and assign feature:",Par[pod].feature)

        # quantities for new tree
        new_ltNum = numLT(Root)
        new_height = getHeight(Root)
        new_nodeNum = getNum(Root)
        newTerm = []  # terminal
        new_nTerm = []  # non-terminal
        newTree = genList(Root)
        for i in np.arange(0, len(newTree)):
            if newTree[i].type == 0:
                newTerm.append(newTree[i])
            else:
                new_nTerm.append(newTree[i])

        # calculate Q
        Q = p_prune / ((len(Nterm) - 1) * n_feature)

        # calculate Qinv (correspond to grow)
        pg = 1 - 0.25 * new_ltNum / (new_ltNum + 3) * 0.75 * min(1, 4 / (len(new_nTerm) + 2))
        Qinv = pg * np.exp(fstrc[0]) / len(newTerm)

    # detransformation
    elif test <= p_stay + p_grow + p_prune + p_detr:
        action = 'detransform'

        det_od = np.random.randint(0, len(detcd), 1)[0]
        det_node = detcd[det_od]
        cutt = None

        # print("cutted op:",det_node.operator)

        Q = p_detr / len(detcd)

        if det_node.parent is None:  # root
            if det_node.right is None:  # one child
                Root = Root.left
            else:  # two children
                if det_node.left.type == 0:  # left is terminal
                    cutt = Root.left
                    Root = Root.right
                elif det_node.right.type == 0:  # right is terminal
                    cutt = Root.right
                    Root = Root.left
                else:  # both are non-terminal
                    aa = np.random.uniform(0, 1, 1)[0]
                    if aa <= 0.5:  # preserve left
                        cutt = Root.right
                        Root = Root.left
                    else:
                        cutt = Root.left
                        Root = Root.right
                    Q = Q / 2
            Root.parent = None
            upDepth(Root)
        else:  # not root, non-terminal
            if det_node.type == 1:  # unary
                if det_node.parent.left is det_node:  # left child of its parent
                    det_node.parent.left = det_node.left
                    det_node.left.parent = det_node.parent
                else:
                    det_node.parent.right = det_node.left
                    det_node.left.parent = det_node.parent
            else:  # binary
                aa = np.random.uniform(0, 1, 1)[0]
                if aa <= 0.5:  # preserve left
                    cutt = det_node.right
                    if det_node.parent.left is det_node:
                        det_node.parent.left = det_node.left
                        det_node.left.parent = det_node.parent
                    else:
                        det_node.parent.right = det_node.left
                        det_node.left.parent = det_node.parent
                else:  # preserve right
                    cutt = det_node.left
                    if det_node.parent.left is det_node:
                        det_node.parent.left = det_node.right
                        det_node.right.parent = det_node.parent
                    else:
                        det_node.parent.right = det_node.right
                        det_node.right.parent = det_node.parent
                Q = Q / 2
            Root.parent = None
            upDepth(Root)

        # the number of linear nodes may decrease
        new_ltnum = 0
        new_tree = genList(Root)
        for i in np.arange(len(new_tree)):
            if new_tree[i].operator == 'ln':
                new_ltnum += 1
        if new_ltnum < ltNum:
            change = 'shrinkage'

        new_pstay = 0.25 * new_ltnum / (new_ltnum + 3)

        # calculate Qinv (correspond to transform)
        new_detcd = []
        for i in np.arange(len(new_tree)):
            flag = True
            if new_tree[i].type == 0:  # terminal is not allowed
                flag = False
            if new_tree[i].parent is None:  # root
                if new_tree[i].right is None and new_tree[i].left.type == 0:
                    flag = False
                elif new_tree[i].left.type == 0 and new_tree[i].right.type == 0:
                    flag = False
            if flag == True:
                new_detcd.append(new_tree[i])
        new_pdetr = (1 - new_pstay) * (1 / 3) * len(new_detcd) / (len(new_detcd) + 3)
        new_ptr = (1 - new_pstay) / 3 - new_pdetr
        Qinv = new_ptr * Op_weights[det_node.op_ind] / len(new_tree)
        if cutt is not None:
            fstrc = fStruc(cutt, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
            Qinv = Qinv * np.exp(fstrc[0])




    # transformation
    elif test <= p_stay + p_grow + p_prune + p_detr + p_trans:
        action = "transform"

        # transform means uniformly pick a node and add an operator as its parent
        # probability of operators is specified by Op_weights
        # adding unary operator is simple
        # adding binary operator needs to generate a new sub-tree
        Tree = genList(Root)
        ins_ind = np.random.randint(0, len(Tree), 1)[0]
        ins_node = Tree[ins_ind]
        ins_opind = np.random.choice(np.arange(0, len(Ops)), p=Op_weights)
        ins_op = Ops[ins_opind]
        ins_type = Op_type[ins_opind]
        ins_opweight = Op_weights[ins_opind]

        # create new node
        new_node = Node(ins_node.depth)
        new_node.operator = ins_op
        new_node.type = ins_type
        new_node.op_ind = ins_opind

        if ins_type == 1:  # unary
            if ins_op == 'ln':  # linear node
                change = 'expansion'
                # new_node.a = norm.rvs(loc=1,scale=np.sqrt(sigma_a))
                # new_node.b = norm.rvs(loc=0,scale=np.sqrt(sigma_b))
            if ins_node.parent is None:  # inserted node is root
                Root = new_node
                new_node.left = ins_node
                ins_node.parent = new_node
            else:  # inserted node is not root
                if ins_node.parent.left is ins_node:
                    ins_node.parent.left = new_node
                else:
                    ins_node.parent.right = new_node
                new_node.parent = ins_node.parent
                new_node.left = ins_node
                ins_node.parent = new_node
            upDepth(Root)
            # calculate Q
            Q = p_trans * ins_opweight / len(Tree)

        else:  # binary
            if ins_node.parent is None:  # inserted node is root
                Root = new_node
                new_node.left = ins_node
                ins_node.parent = new_node
                new_right = Node(1)
                new_node.right = new_right
                new_right.parent = new_node
                upDepth(Root)
                grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                fstrc = fStruc(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                # calculate Q
                Q = p_trans * ins_opweight * np.exp(fstrc[0]) / len(Tree)
            else:  # inserted node is not root
                # place the new node
                if ins_node.parent.left is ins_node:
                    ins_node.parent.left = new_node
                    new_node.parent = ins_node.parent
                else:
                    ins_node.parent.right = new_node
                    new_node.parent = ins_node.parent

                new_node.left = ins_node
                ins_node.parent = new_node
                new_right = Node(new_node.depth + 1)
                new_node.right = new_right
                new_right.parent = new_node
                upDepth(Root)
                grow(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                fstrc = fStruc(new_right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

                # calculate Q
                Q = p_trans * ins_opweight * np.exp(fstrc[0]) / len(Tree)

        # the number of linear nodes may decrease
        new_ltnum = 0
        new_tree = genList(Root)
        for i in np.arange(len(new_tree)):
            if new_tree[i].operator == 'ln':
                new_ltnum += 1
        if new_ltnum > ltNum:
            change = 'expansion'

        # calculate Qinv (correspond to detransform)
        new_pstay = 0.25 * new_ltnum / (new_ltnum + 3)

        new_detcd = []
        for i in np.arange(len(new_tree)):
            flag = True
            if new_tree[i].type == 0:  # terminal is not allowed
                flag = False
            if new_tree[i].parent is None:  # root
                if new_tree[i].right is None and new_tree[i].left.type == 0:
                    flag = False
                elif new_tree[i].left.type == 0 and new_tree[i].right.type == 0:
                    flag = False
            if flag == True:
                new_detcd.append(new_tree[i])

        new_pdetr = (1 - new_pstay) * (1 / 3) * len(new_detcd) / (len(new_detcd) + 3)
        new_ptr = (1 - new_pstay) / 3 - new_pdetr

        Qinv = new_pdetr / len(new_detcd)
        if new_node.type == 2:
            if new_node.left.type > 0 and new_node.right.type > 0:
                Qinv = Qinv / 2



    # reassignOperator
    elif test <= p_stay + p_grow + p_prune + p_detr + p_trans + p_rop:
        action = 'ReassignOperator'
        # print("action:",action)
        pod = np.random.randint(0, len(Nterm), 1)[0]
        last_op = Nterm[pod].operator
        last_op_ind = Nterm[pod].op_ind
        last_type = Nterm[pod].type
        # print('replaced operator:',last_op)
        cnode = Nterm[pod]  ########pointer to the node changed#######
        # a deep copy of the changed node and its descendents
        replaced = copy.deepcopy(Nterm[pod])

        new_od = np.random.choice(np.arange(0, len(Ops)), p=Op_weights)
        new_op = Ops[new_od]
        # print('assign new operator:',new_op)
        new_type = Op_type[new_od]

        # originally unary
        if last_type == 1:
            if new_type == 1:  # unary to unary
                # assign operator and type
                Nterm[pod].operator = new_op
                if last_op == 'ln':  # originally linear
                    if new_op != 'ln':  # change from linear to other ops
                        cnode.a = None
                        cnode.b = None
                        change = 'shrinkage'
                else:  # orignally not linear
                    if new_op == 'ln':  # linear increases by 1
                        ###### a and b is not sampled
                        change = 'expansion'

                # calculate Q, Qinv (equal)
                Q = Op_weights[new_od]
                Qinv = Op_weights[last_op_ind]

            else:  # unary to binary
                # assign operator and type
                cnode.operator = new_op
                cnode.type = 2
                if last_op == 'ln':
                    cnode.a = None
                    cnode.b = None
                    # grow a new sub-tree rooted at right child
                cnode.right = Node(cnode.depth + 1)
                cnode.right.parent = cnode
                grow(cnode.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                fstrc = fStruc(cnode.right, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)

                # calculate Q
                Q = p_rop * np.exp(fstrc[0]) * Op_weights[new_od] / (len(Nterm))
                # calculate Qinv
                # get necessary quantities
                new_height = getHeight(Root)
                new_nodeNum = getNum(Root)
                newTerm = []  # terminal
                newTree = genList(Root)
                new_ltNum = numLT(Root)
                for i in np.arange(0, len(newTree)):
                    if newTree[i].type == 0:
                        newTerm.append(newTree[i])
                    # reversed action is binary to unary
                new_p0 = new_ltNum / (4 * (new_ltNum + 3))
                Qinv = 0.125 * (1 - new_p0) * Op_weights[last_op_ind] / (new_nodeNum - len(newTerm))

                # record change of dim
                if new_ltNum > ltNum:
                    change = 'expansion'
                elif new_ltNum < ltNum:
                    change = 'shrinkage'




        # originally binary
        else:
            if new_type == 1:  # binary to unary
                # assign operator and type
                cutted = copy.deepcopy(cnode.right)  # deep copy root of the cutted subtree
                # preserve pointers to all cutted ln
                p_ltNum = numLT(cutted)
                if p_ltNum > 1:
                    change = 'shrinkage'
                elif new_op == 'ln':
                    if p_ltNum == 0:
                        change = 'expansion'

                cnode.right = None
                cnode.operator = new_op
                cnode.type = new_type

                # calculate Q
                Q = p_rop * Op_weights[new_od] / len(Nterm)
                # calculate Qinv
                # necessary quantities
                new_nodeNum = getNum(Root)
                newTerm = []  # terminal
                newTree = genList(Root)
                new_ltNum = numLT(Root)
                # reversed action is unary to binary and grow
                new_p0 = new_ltNum / (4 * (new_ltNum + 3))
                fstrc = fStruc(cutted, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
                Qinv = 0.125 * (1 - new_p0) * np.exp(fstrc[0]) * Op_weights[last_op_ind] / (
                (new_nodeNum - len(newTerm)))



            else:  # binary to binary
                # assign operator
                cnode.operator = new_op
                # calculate Q,Qinv(equal)
                Q = Op_weights[new_od]
                Qinv = Op_weights[last_op_ind]


    # reassign feature
    else:
        action = 'ReassignFeature'
        # print("action:",action)

        # pick a terminal node
        pod = np.random.randint(0, len(Term), 1)[0]
        # pick a feature and reassign
        fod = np.random.randint(0, n_feature, 1)
        Term[pod].feature = fod
        # calculate Q,Qinv (equal)
        Q = Qinv = 1

    Root.parent = None
    upDepth(Root)
    # print(action)

    return [oldRoot, Root, lnPointers, change, Q, Qinv, last_a, last_b, cnode]


# =============================================================================
# # calculate the likelihood of genenerating auxiliary variable
# # change is a string with value of 'expansion' or 'shrinkage'
# # oldRoot is the root of the original Tree
# # Root is the root of the new tree
# # cnode is a pointer to the node just changed (expand or shrink 'ln') (if applicable)
# # last_a and last_b is the parameters for the shrinked node (if applicable)
# # lnPointers is list of pointers to originally linear nodes in Tree before changing
# =============================================================================
def auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a, last_b, cnode=None):
    # record the informations of linear nodes other than the shrinked or expanded
    odList = []  # list of orders of linear nodes
    Tree = genList(Root)

    for i in np.arange(0, len(Tree)):
        if Tree[i].operator == 'ln':
            odList.append(i)

    # sample new sigma_a2 and sigma_b2
    new_sa2 = invgamma.rvs(1)
    new_sb2 = invgamma.rvs(1)
    old_sa2 = sigma_a
    old_sb2 = sigma_b

    if change == 'shrinkage':
        prsv_aList = []
        prsv_bList = []
        cut_aList = []
        cut_bList = []
        # find the preserved a's
        for i in np.arange(0, len(lnPointers)):
            if lnPointers[i].operator == 'ln':  # still linear
                prsv_aList.append(last_a[i])
                prsv_bList.append(last_b[i])
            else:  # no longer linear
                cut_aList.append(last_a[i])
                cut_bList.append(last_b[i])
        # substitute those cutted with newly added if cut and add
        for i in np.arange(0, len(odList) - len(prsv_aList)):
            prsv_aList.append(cut_aList[i])
            prsv_bList.append(cut_bList[i])

        n0 = len(prsv_aList)

        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0, n0):
            UaList.append(norm.rvs(loc=0, scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # generate inverse auxiliary U*
        NaList = []  # Theta* with a
        NbList = []  # Theta* with b
        NUaList = []  # U* with a
        NUbList = []  # U* with b
        for i in np.arange(0, n0):
            NaList.append(prsv_aList[i] + UaList[i])
            NbList.append(prsv_bList[i] + UbList[i])
            NUaList.append(prsv_aList[i] - UaList[i])
            NUbList.append(prsv_bList[i] - UbList[i])
        NUaList = NUaList + last_a
        NUbList = NUbList + last_b

        # hstar is h(U*|Theta*,S*,S^t) (here we calculate the log) corresponding the shorter para
        # h is h(U|Theta,S^t,S*) corresponding longer para
        # Theta* is the
        logh = 0
        loghstar = 0

        # contribution of new_sa2 and new_sb2
        logh += np.log(invgamma.pdf(new_sa2, 1))
        logh += np.log(invgamma.pdf(new_sb2, 1))
        loghstar += np.log(invgamma.pdf(old_sa2, 1))
        loghstar += np.log(invgamma.pdf(old_sb2, 1))

        for i in np.arange(0, len(UaList)):
            # contribution of UaList and UbList
            logh += np.log(norm.pdf(UaList[i], loc=0, scale=np.sqrt(new_sa2)))
            logh += np.log(norm.pdf(UbList[i], loc=0, scale=np.sqrt(new_sb2)))

        for i in np.arange(0, len(NUaList)):
            # contribution of NUaList and NUbList
            loghstar += np.log(norm.pdf(NUaList[i], loc=0, scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i], loc=0, scale=np.sqrt(old_sb2)))

        hratio = np.exp(loghstar - logh)
        # print("hratio:",hratio)

        # determinant of jacobian
        detjacob = np.power(2, 2 * len(prsv_aList))
        # print("detjacob:",detjacob)

        #### assign Theta* to the variables
        # new values of Theta
        # new sigma_a, sigma_b are directly returned
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [hratio, detjacob, new_sa2, new_sb2]



    elif change == 'expansion':
        # sample new sigma_a2 and sigma_b2
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        old_sa2 = sigma_a
        old_sb2 = sigma_b

        # lists of theta_0 and expanded ones
        # last_a is the list of all original a's
        # last_b is the list of all original b's
        odList = []
        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)

        # sample auxiliary U
        UaList = []
        UbList = []
        for i in np.arange(0, len(last_a)):
            UaList.append(norm.rvs(loc=0, scale=np.sqrt(new_sa2)))
            UbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # generate inverse auxiliary U* and new para Theta*
        NaList = []  # Theta*_a
        NbList = []  # Theta*_b
        NUaList = []  # U*_a
        NUbList = []  # U*_b
        for i in np.arange(0, len(last_a)):
            NaList.append((last_a[i] + UaList[i]) / 2)
            NbList.append((last_b[i] + UbList[i]) / 2)
            NUaList.append((last_a[i] - UaList[i]) / 2)
            NUbList.append((last_b[i] - UbList[i]) / 2)

        # append newly generated a and b into NaList and NbList
        nn = len(odList) - len(last_a)  # number of newly added ln
        for i in np.arange(0, nn):
            u_a = norm.rvs(loc=1, scale=np.sqrt(new_sa2))
            u_b = norm.rvs(loc=0, scale=np.sqrt(new_sb2))
            NaList.append(u_a)
            NbList.append(u_b)

        # calculate h ratio
        # logh is h(U|Theta,S,S*) correspond to jump from short to long (new)
        # loghstar is h(Ustar|Theta*,S*,S) correspond to jump from long to short
        logh = 0
        loghstar = 0

        # contribution of sigma_ab
        logh += np.log(invgamma.pdf(new_sa2, 1))
        logh += np.log(invgamma.pdf(new_sb2, 1))
        loghstar += np.log(invgamma.pdf(old_sa2, 1))
        loghstar += np.log(invgamma.pdf(old_sb2, 1))

        # contribution of u_a, u_b
        for i in np.arange(len(last_a), nn):
            logh += norm.pdf(NaList[i], loc=1, scale=np.sqrt(new_sa2))
            logh += norm.pdf(NbList[i], loc=0, scale=np.sqrt(new_sb2))

        # contribution of U_theta
        for i in np.arange(0, len(UaList)):
            logh += np.log(norm.pdf(UaList[i], loc=0, scale=np.sqrt(new_sa2)))
            logh += np.log(norm.pdf(UbList[i], loc=0, scale=np.sqrt(new_sb2)))

        for i in np.arange(0, len(NUaList)):
            loghstar += np.log(norm.pdf(NUaList[i], loc=0, scale=np.sqrt(old_sa2)))
            loghstar += np.log(norm.pdf(NUbList[i], loc=0, scale=np.sqrt(old_sb2)))

        # compute h ratio
        hratio = np.exp(loghstar - logh)

        # determinant of jacobian
        detjacob = 1 / np.power(2, 2 * len(last_a))

        #### assign Theta* to the variables
        # new values of sigma_a sigma_b
        # new values of Theta
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [hratio, detjacob, new_sa2, new_sb2]


    else:  # same set of parameters
        # record the informations of linear nodes other than the shrinked or expanded
        odList = []  # list of orders of linear nodes
        Tree = genList(Root)

        old_sa2 = sigma_a
        old_sb2 = sigma_b

        for i in np.arange(0, len(Tree)):
            if Tree[i].operator == 'ln':
                odList.append(i)

        NaList = []
        NbList = []
        new_sa2 = invgamma.rvs(1)
        new_sb2 = invgamma.rvs(1)
        for i in np.arange(0, len(odList)):
            NaList.append(norm.rvs(loc=1, scale=np.sqrt(new_sa2)))
            NbList.append(norm.rvs(loc=0, scale=np.sqrt(new_sb2)))

        # new values of Theta
        for i in np.arange(0, len(odList)):
            Tree[odList[i]].a = NaList[i]
            Tree[odList[i]].b = NbList[i]

        return [new_sa2, new_sb2]


# =============================================================================
# # calculate the log likelihood f(y|S,Theta,x)
# # (S,Theta) is represented by node Root
# # prior is y ~ N(output,sigma)
# # output is the matrix of outputs corresponding to different roots
# =============================================================================
def ylogLike(y, outputs, sigma):
    XX = copy.deepcopy(outputs)
    scale = np.max(np.abs(XX))
    XX = XX / scale
    epsilon = np.eye(XX.shape[1])*1e-6
    yy = np.array(y)
    yy.shape = (yy.shape[0], 1)
    Beta = np.linalg.inv(np.matmul(XX.transpose(), XX)+epsilon)
    Beta = np.matmul(Beta, np.matmul(XX.transpose(), yy))

    output = np.matmul(XX, Beta)

    # error = 0
    # for i in np.arange(0, len(y)):
    #     error += (y[i] - output[i, 0]) * (y[i] - output[i, 0])
    error = np.sum(np.square(y-output[:,0]))
    #error = np.sqrt(error / len(y))
    # print("mean error:",error)

    # log_sum = 0
    # for i in np.arange(0, len(y)):
    #     temp = np.power(y[i] - output[i, 0], 2)  # np.log(norm.pdf(y[i],loc=output[i,0],scale=np.sqrt(sigma)))
    #     # print(i,temp)
    #     log_sum += temp
    log_sum = error
    log_sum = -log_sum / (2 * sigma * sigma)
    log_sum -= 0.5 * len(y) * np.log(2 * np.pi * sigma * sigma)
    return (log_sum)


# =============================================================================
# # prop new structure, sample new parameters and decide whether to accept
# # Root is the root node of the tree to be changed
# # RootLists stores the list of roots of K trees
# # sigma is for output to y
# # sigma_a, sigma_b are (squared) hyper-paras for linear paras
# =============================================================================
def newProp(Roots, count, sigma, y, indata, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b):
    # number of components
    K = len(Roots)
    # the root to edit
    Root = copy.deepcopy(Roots[count])
    [oldRoot, Root, lnPointers, change, Q, Qinv, last_a, last_b, cnode] = Prop(Root, n_feature, Ops, Op_weights,
                                                                               Op_type, beta, sigma_a, sigma_b)
    # print("change:",change)
    # display(genList(Root))
    # allcal(Root,train_data)
    sig = 4
    new_sigma = invgamma.rvs(sig)

    # matrix of outputs
    new_outputs = np.zeros((len(y), K))
    old_outputs = np.zeros((len(y), K))

    # auxiliary propose
    if change == 'shrinkage':
        [hratio, detjacob, new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a,
                                                       last_b, cnode)
    elif change == 'expansion':
        [hratio, detjacob, new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a,
                                                       last_b, cnode)
    else:  # no dimension jump
        # the parameters are upgraded as well
        [new_sa2, new_sb2] = auxProp(change, oldRoot, Root, lnPointers, sigma_a, sigma_b, last_a, last_b, cnode)

    for i in np.arange(K):
        if i == count:
            temp = allcal(Root, indata)
            temp.shape = (temp.shape[0])
            new_outputs[:, i] = temp
            temp = allcal(oldRoot, indata)
            temp.shape = (temp.shape[0])
            old_outputs[:, i] = temp
        else:
            temp = allcal(Roots[i], indata)
            temp.shape = (temp.shape[0])
            new_outputs[:, i] = temp
            old_outputs[:, i] = temp

    if np.linalg.matrix_rank(new_outputs) < K:
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]

    if change == 'shrinkage':
        # contribution of f(y|S,Theta,x)
        # print("new sigma:",round(new_sigma,3))
        yllstar = ylogLike(y, new_outputs, new_sigma)
        # print("sigma:",round(sigma,3))
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = yllstar - yll
        # print("log yratio:",log_yratio)

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2)
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        sl = strucl[0] + strucl[1]
        slstar = struclstar[0] + struclstar[1]
        log_strucratio = slstar - sl  # struclstar / strucl
        # print("log strucratio:",log_strucratio)

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))
        # print("log qratio:",log_qratio)

        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(max(1e-5,hratio)) + np.log(max(1e-5,detjacob))
        logR = logR + np.log(invgamma.pdf(new_sigma, sig)) - np.log(invgamma.pdf(sigma, sig))
        # print("logR:",logR)

    elif change == 'expansion':
        # contribution of f(y|S,Theta,x)
        yllstar = ylogLike(y, new_outputs, new_sigma)
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = yllstar - yll

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2)
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)
        sl = strucl[0] + strucl[1]
        slstar = struclstar[0] + struclstar[1]
        log_strucratio = slstar - sl  # struclstar / strucl

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))

        # R
        logR = log_yratio + log_strucratio + log_qratio + np.log(max(1e-5,hratio)) + np.log(max(1e-5,detjacob))
        logR = logR + np.log(invgamma.pdf(new_sigma, sig)) - np.log(invgamma.pdf(sigma, sig))

    else:  # no dimension jump
        # contribution of f(y|S,Theta,x)
        yllstar = ylogLike(y, new_outputs, new_sigma)
        yll = ylogLike(y, old_outputs, sigma)

        log_yratio = yllstar - yll
        # yratio = np.exp(yllstar-yll)

        # contribution of f(Theta,S)
        strucl = fStruc(Root, n_feature, Ops, Op_weights, Op_type, beta, new_sa2, new_sb2)[0]
        struclstar = fStruc(oldRoot, n_feature, Ops, Op_weights, Op_type, beta, sigma_a, sigma_b)[0]
        log_strucratio = struclstar - strucl

        # contribution of proposal Q and Qinv
        log_qratio = np.log(max(1e-5,Qinv / Q))

        # R
        logR = log_yratio + log_strucratio + log_qratio
        logR = logR + np.log(invgamma.pdf(new_sigma, sig)) - np.log(invgamma.pdf(sigma, sig))

    alpha = min(logR, 0)
    test = np.random.uniform(low=0, high=1, size=1)[0]
    if np.log(test) >= alpha:  # no accept
        # print("no accept")
        Root = oldRoot
        return [False, sigma, copy.deepcopy(oldRoot), sigma_a, sigma_b]
    else:
        # print("||||||accepted||||||")
        return [True, new_sigma, copy.deepcopy(Root), new_sa2, new_sb2]




