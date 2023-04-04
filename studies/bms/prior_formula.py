import os
from prior_frequencies import get_raw_priors
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# P_1 = a_1 * np + a_2 * nv + b * log(frequency) + c_i
# a_1 = 0.0388, a_2 = 0.00649, b = -1.34, c_i = 0.00661
# P_2 = a_1 * np + a_2 * nv + a * log(np + nv) + b * log(frequency) + c_i + is_/
# a_1 = -0.0514, a_2 = -0.0738, a = 1.97, b = -1.36, c_i = 0.00153, is_/ = -0.0963


def get_priors(prior_name, num_param, num_var, refit=False):
    op_freq, op_type = get_raw_priors(prior_name)
    prior = freq_to_prior(op_freq, op_type, num_param, num_var, refit)
    return prior


def freq_to_prior(freq, op_type, num_param, num_var, prior_name, refit=False):
    if refit:
        params_1 = fit_formula(prior_name, 1)
        params_2 = fit_formula(prior_name, 2)
    else:
        params_1 = [0.00661, 0.0388, 0.00649, -1.34]
        params_2 = [0.00154, -0.0513, -0.0737, -1.36, 1.97, -0.0963]
    prior = {}
    for op in freq:
        if op_type[op] == 1:
            value = params_1[0] + params_1[1] * num_param + params_1[2] * num_var + \
                    params_1[3] * np.log(freq[op])
        elif op_type[op] == 2:
            value = params_2[0] + params_2[1] * num_param + params_2[2] * num_var + \
                    params_2[3] * np.log(freq[op]) + params_2[4] * np.log(num_param + num_var)
            if op == '/':  # additional constant for the one non-commutative 2-operand operator
                value += params_2[5]
        else:
            raise KeyError('This prior is not available')
        prior.update({op: value})
    return prior


def read_prior_par(filename):
    with open(filename) as inf:
        lines = inf.readlines()
    par = dict(list(zip(lines[0].strip().split()[1:],
                    [float(x) for x in lines[-1].strip().split()[1:]])))
    return par


def extract_info(file_name):
    nv_start = file_name.index('nv')
    nv_stop = file_name.index('.', nv_start)
    np_start = file_name.index('np')
    np_stop = file_name.index('.', np_start)
    nv = int(file_name[nv_start+2:nv_stop])
    np = int(file_name[np_start + 2:np_stop])
    return nv, np


def get_prior_dicts(op_type):
    prior_dicts = []
    idx = 0
    for file in os.listdir('Prior/'):
        # print(file)
        if file.endswith('.dat'):
            [num_var, num_param] = extract_info(file)
            if num_var < 25 and num_param < 25:
                prior = read_prior_par('Prior/' + file)
                new_dict = dict()
                new_dict.update({'nv': num_var, 'np': num_param})
                idx += 2
                prior_ops = prior.keys()
                ops = get_raw_priors('Default')[0].keys()
                # print(prior_ops)
                # print(ops)
                raw = get_raw_priors()[1]
                # print(raw)
                for op in ops:
                    if ('Nopi_'+op) in prior_ops:
                        if raw[op] == op_type:
                            new_dict.update({op: prior['Nopi_'+op]})
                            idx += 1
                prior_dicts.append(new_dict)
    return prior_dicts, idx


def fit_formula(raw_prior_name, op_type):
    dics, idx = get_prior_dicts(op_type=op_type)
    raw, raw_type = get_raw_priors(raw_prior_name)
    dic_length = len(dics[0])
    if op_type == 1:
        data = np.zeros((idx * dic_length, 4))
        idx = 0
        for dic in dics:
            for op in dic.keys():
                if op not in ['nv', 'np']:
                    new_line = [dic['np'], dic['nv'], np.log(raw[op]), dic[op]]
                    data[idx] = new_line
                    idx += 1
        a = 0
        b = 0
        c = 0
        d = 0
        score = 0
        repetitions = 1000
        for rep in range(repetitions):
            X_train, X_test, y_train, y_test = train_test_split(data[:, 0:3], data[:, 3], test_size=0.2)
            model = LinearRegression().fit(X_train, y_train)
            a += model.intercept_ / repetitions
            b += model.coef_[0] / repetitions
            c += model.coef_[1] / repetitions
            d += model.coef_[2] / repetitions
            score += model.score(X_test, y_test) / repetitions
        return [a, b, c, d]
    elif op_type == 2:
        data = np.zeros((idx * dic_length, 6))
        idx = 0
        for dic in dics:
            for op in dic.keys():
                if op not in ['nv', 'np']:
                    new_line = [dic['np'], dic['nv'], np.log(raw[op]), np.log(dic['np'] + dic['nv']),
                                op == '/', dic[op]]
                    data[idx] = new_line
                    idx += 1
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        f = 0
        score = 0
        repetitions = 1000
        for rep in range(repetitions):
            X_train, X_test, y_train, y_test = train_test_split(data[:, 0:5], data[:, 5], test_size=0.2)
            model = LinearRegression().fit(X_train, y_train)
            a += model.intercept_ / repetitions
            b += model.coef_[0] / repetitions
            c += model.coef_[1] / repetitions
            d += model.coef_[2] / repetitions
            e += model.coef_[3] / repetitions
            f += model.coef_[4] / repetitions
            score += model.score(X_test, y_test) / repetitions
        return [a, b, c, d, e, f]


if __name__ == '__main__':
    dics, idx = get_prior_dicts(op_type=1)
    print(idx-141*2)
    raw, raw_type = get_raw_priors('Guimera2020')
    dic_length = len(dics[0])
    # np, nv, log(freq), log(np+nv), prior
    data = np.zeros((idx*dic_length, 5))
    idx = 0
    for dic in dics:
        for op in dic.keys():
            if op not in ['nv', 'np']:
                new_line = [dic['np'], dic['nv'], np.log(raw[op]), np.log(dic['np']+dic['nv']), dic[op]]
                data[idx] = new_line
                idx += 1
    a = 0
    b = 0
    c = 0
    d = 0
    e = 0
    score = 0
    repetitions = 1000
    for rep in range(repetitions):
        X_train, X_test, y_train, y_test = train_test_split(data[:, 0:4], data[:, 4], test_size=0.2)
        model = LinearRegression().fit(X_train, y_train)
        a += model.intercept_/repetitions
        b += model.coef_[0]/repetitions
        c += model.coef_[1]/repetitions
        d += model.coef_[2]/repetitions
        e += model.coef_[3]/repetitions
        score += model.score(X_test, y_test)/repetitions
    print((a, b, c, d, e))
    print(score)
