import time
import json
import os, sys

sys.path.append('../algorithms/tpsr/')
sys.path.append('../algorithms/tpsr/TPSR')
sys.path.append('../algorithms/e2et/symbolicregression/')

import torch
import numpy as np

from TPSR.nesymres.src.nesymres.architectures.model import Model
from TPSR.nesymres.src.nesymres.dclasses import FitParams, BFGSParams
from functools import partial
import sympy as sp
from sympy import lambdify
import omegaconf

from sklearn.base import BaseEstimator, RegressorMixin

script_dir = os.path.dirname(os.path.realpath(__file__))
eq_setting = os.path.join(script_dir, '../../../algorithms/tpsr/TPSR/nesymres/jupyter/100M/eq_setting.json')
with open(eq_setting, 'r') as json_file:
    eq_setting = json.load(json_file)

config = os.path.join(script_dir, "../../../algorithms/tpsr/TPSR/nesymres/jupyter/100M/config.yaml")
cfg = omegaconf.OmegaConf.load(config)

def get_variables(equation):
    """ Parse all free variables in equations and return them in
    lexicographic order"""

    expr = sp.parse_expr(equation)
    variables = expr.free_symbols
    variables = {str(v) for v in variables}

    # # Tighter sanity check: we only accept variables in ascending order
    # # to avoid silent errors with lambdify later.
    # if (variables not in [{'x'}, {'x', 'y'}, {'x', 'y', 'z'}]
    #         and variables not in [{'x1'}, {'x1', 'x2'}, {'x1', 'x2', 'x3'}]):
    #     raise ValueError(f'Unexpected set of variables: {variables}. '
    #                      f'If you want to allow this, make sure that the '
    #                      f'order of variables of the lambdify output will be '
    #                      f'correct.')

    # Make a sorted list out of variables
    # Assumption: the correct order is lexicographic (x, y, z)
    variables = sorted(variables)

    return variables


def evaluate_func(func_str, vars_list, X):
    assert X.ndim == 2
    assert len(set(vars_list)) == len(vars_list), 'Duplicates in vars_list!'

    order_list = vars_list
    indeces = [int(x[2:])-1 for x in order_list]

    if not order_list:
        # Empty order list. Constant function predicted
        f = lambdify([], func_str)
        return f() * np.ones(X.shape[0])

    # Pad X with zero-columns, allowing for variables to appear in the equation
    # that are not in the ground-truth equation
    X_padded = np.zeros((X.shape[0], len(vars_list)))

    
    X_padded[:, :X.shape[1]] = X[:,:X_padded.shape[1]]
    # Subselect columns of X that corrspond to provided variables
    X_subsel = X_padded[:, indeces]

    # The positional arguments of the resulting function will correspond to
    # the order of variables in "vars_list"
    f = lambdify(vars_list, func_str)
    return f(*X_subsel.T)


class NeSymResRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        #Main
        ## Set up BFGS load rom the hydra config yaml
        bfgs = BFGSParams(
            activated= cfg.inference.bfgs.activated,
            n_restarts=cfg.inference.bfgs.n_restarts,
            add_coefficients_if_not_existing=cfg.inference.bfgs.add_coefficients_if_not_existing,
            normalization_o=cfg.inference.bfgs.normalization_o,
            idx_remove=cfg.inference.bfgs.idx_remove,
            normalization_type=cfg.inference.bfgs.normalization_type,
            stop_time=cfg.inference.bfgs.stop_time,
        )

        eq_setting["total_variables"] = [f"x_{i+1}" for i in range(X.shape[1])]
        eq_setting["num_variables"] = int(X.shape[1])
        eq_setting["variables"] = [f"x_{i+1}" for i in range(X.shape[1])]

        print(eq_setting["total_variables"])

        params_fit = FitParams(word2id=eq_setting["word2id"], 
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                una_ops=eq_setting["una_ops"], 
                                bin_ops=eq_setting["bin_ops"], 
                                total_variables=eq_setting["total_variables"],  
                                total_coefficients=eq_setting["total_coefficients"],
                                rewrite_functions=eq_setting["rewrite_functions"],
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

        weights_path = os.path.join(script_dir, "../../../algorithms/tpsr/TPSR/nesymres/weights/10M.ckpt")

        ## Load architecture, set into eval mode, and pass the config parameters
        #cfg.architecture.num_features = eq_setting["num_variables"]
        
        model = Model.load_from_checkpoint(weights_path, cfg=cfg.architecture)
        model.eval()

        if torch.cuda.is_available(): 
            model.cuda()

        fitfunc = partial(model.fitfunc, cfg_params=params_fit)

        output_ref = fitfunc(X,y) 

        self.model_skeleton_ = output_ref
        self.model_eq_       = model.get_equation()

        print("NeSymReS output ref: ", output_ref)
        print("NeSymReS Equation Skeleton: ", self.model_skeleton_)
        print("NeSymReS model_eq: ", self.model_eq_)

        self.pred_variables_ = get_variables(self.model_eq_[0])

        return self
    

    def predict(self, X):
        return evaluate_func(self.model_eq_[0],
                             self.pred_variables_, X)


est = NeSymResRegressor()

def model(est, X=None):
    return str(est.model_eq_[0])
