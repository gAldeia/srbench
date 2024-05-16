import time
import json
import os, sys

sys.path.append('../tpsr/')
sys.path.append('../tpsr/TPSR')

import torch
import numpy as np

from TPSR.nesymres.src.nesymres.architectures.model import Model
from TPSR.nesymres.src.nesymres.dclasses import FitParams, BFGSParams
from functools import partial
import sympy as sp
from sympy import lambdify
import omegaconf

from sklearn.base import BaseEstimator, RegressorMixin


# Hardcoding this (because we dont want to mess with parse)
class NESYMRES_params():
    def __init__(self):
        self.ablation_to_keep=None
        self.accumulate_gradients=1
        self.actor_lr=1e-06
        self.alg='mcts'
        self.amp=-1
        self.attention_dropout=0
        self.backbone_model='e2e'
        self.batch_load=False
        self.batch_size=1
        self.batch_size_eval=64
        self.beam_early_stopping=True
        self.beam_eval=True
        self.beam_eval_train=0
        self.beam_length_penalty=1
        self.beam_selection_metrics=1
        self.beam_size=1
        self.beam_temperature=0.1
        self.beam_type='sampling'
        self.clip_grad_norm=0.5
        self.collate_queue_size=2000
        self.cpu=True
        self.critic_lr=1e-05
        self.debug=False
        self.debug_slurm=False
        self.debug_train_statistics=False
        self.dec_emb_dim=512
        self.dec_positional_embeddings='learnable'
        self.device='cpu'
        self.dropout=0
        self.dump_path=''
        self.emb_emb_dim=64
        self.emb_expansion_factor=1
        self.embedder_type='LinearPoint'
        self.enc_emb_dim=512
        self.enc_positional_embeddings=None
        self.enforce_dim=True
        self.entropy_coef=0.01
        self.entropy_weighted_strategy='none'
        self.env_base_seed=0
        self.env_name='functions'
        self.eval_data=''
        self.eval_dump_path=None
        self.eval_from_exp=''
        self.eval_in_domain=False
        self.eval_input_length_modulo=-1
        self.eval_mcts_in_domain=False
        self.eval_mcts_on_pmlb=False
        self.eval_noise=0
        self.eval_noise_gamma=0.0
        self.eval_noise_type='additive'
        self.eval_on_pmlb=False
        self.eval_only=False
        self.eval_size=10000
        self.eval_verbose=0
        self.eval_verbose_print=False
        self.exp_id=''
        self.exp_name='debug'
        self.export_data=False
        self.extra_binary_operators=''
        self.extra_constants=None
        self.extra_unary_operators=''
        self.float_precision=3
        self.fp16=False
        self.gpu_to_use='0'
        self.horizon=200
        self.kl_coef=0.01
        self.kl_regularizer=0.001
        self.lam=0.1
        self.local_rank=-1
        self.lr=1e-05
        self.lr_patience=100
        self.mantissa_len=1
        self.master_port=-1
        self.max_binary_ops_offset=4
        self.max_binary_ops_per_dim=1
        self.max_centroids=10
        self.max_epoch=100000
        self.max_exponent=100
        self.max_exponent_prefactor=1
        self.max_generated_output_len=200
        self.max_input_dimension=10
        self.max_input_points=200
        self.max_int=10
        self.max_len=200
        self.max_number_bags=10
        self.max_output_dimension=1
        self.max_src_len=200
        self.max_target_len=200
        self.max_token_len=0
        self.max_trials=1
        self.max_unary_depth=6
        self.max_unary_ops=4
        self.min_binary_ops_per_dim=0
        self.min_input_dimension=1
        self.min_len_per_dim=5
        self.min_op_prob=0.01
        self.min_output_dimension=1
        self.min_unary_ops=0
        self.n_dec_heads=16
        self.n_dec_hidden_layers=1
        self.n_dec_layers=16
        self.n_emb_layers=1
        self.n_enc_heads=16
        self.n_enc_hidden_layers=1
        self.n_enc_layers=2
        self.n_prediction_points=200
        self.n_steps_per_epoch=3000
        self.n_trees_to_refine=10
        self.n_words=10292
        self.no_prefix_cache=True
        self.no_seq_cache=True
        self.norm_attention=False
        self.num_beams=1
        self.num_workers=10
        self.nvidia_apex=False
        self.operators_to_downsample='div_0,arcsin_0,arccos_0,tan_0.2,arctan_0.2,sqrt_5,pow2_3,inv_3'
        self.operators_to_not_repeat=''
        self.optimizer='adam_inverse_sqrt,warmup_updates=10000'
        self.pad_to_max_dim=True
        self.pmlb_data_type='feynman'
        self.prediction_sigmas='1,2,4,8,16'
        self.print_freq=100
        self.prob_const=0.0
        self.prob_rand=0.0
        self.queue_strategy=None
        self.reduce_num_constants=True
        self.refinements_types='method=BFGS_batchsize=256_metric=/_mse'
        self.reload_checkpoint=''
        self.reload_data=''
        self.reload_model=''
        self.reload_size=-1
        self.required_operators=''
        self.rescale=True
        self.reward_coef=1
        self.reward_type='nmse'
        self.rl_alg='ppo'
        self.rollout=3
        self.run_id=1
        self.sample_only=False
        self.save_eval_dic='./eval_result'
        self.save_model=True
        self.save_periodic=25
        self.save_results=True
        self.seed=23
        self.share_inout_emb=True
        self.simplify=False
        self.stopping_criterion=''
        self.target_kl=1
        self.target_noise=0.0
        self.tasks='functions'
        self.test_env_seed=1
        self.tokens_per_batch=10000
        self.train_noise_gamma=0.0
        self.train_value=False
        self.ts_mode='best'
        self.ucb_base=10.0
        self.ucb_constant=1.0
        self.uct_alg='uct'
        self.update_modules='all'
        self.use_abs=False
        self.use_controller=True
        self.use_skeleton=False
        self.use_sympy=False
        self.validation_metrics='r2_zero,r2,accuracy_l1_biggio,accuracy_l1_1e-3,accuracy_l1_1e-2,accuracy_l1_1e-1,_complexity'
        self.vf_coef=0.0001
        self.warmup_epoch=5
        self.width=3
        self.windows=False

nesymres_params = NESYMRES_params()

np.random.seed(nesymres_params.seed)
torch.manual_seed(nesymres_params.seed)
torch.cuda.manual_seed(nesymres_params.seed)

nesymres_params.cpu    = False if torch.cuda.is_available() else True
nesymres_params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nesymres_params.no_seq_cache=True
nesymres_params.no_prefix_cache=True
nesymres_params.backbone_model="nesymres"

script_dir = os.path.dirname(os.path.realpath(__file__))
eq_setting = os.path.join(script_dir, '../tpsr/TPSR/nesymres/jupyter/100M/eq_setting.json')
with open(eq_setting, 'r') as json_file:
    eq_setting = json.load(json_file)

config = os.path.join(script_dir, "../tpsr/TPSR/nesymres/jupyter/100M/config.yaml")
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
        eq_setting["variables"] = [f"x_{i+1}" for i in range(X.shape[1])]

        params_fit = FitParams(word2id=eq_setting["word2id"], 
                                id2word={int(k): v for k,v in eq_setting["id2word"].items()}, 
                                una_ops=eq_setting["una_ops"], 
                                bin_ops=eq_setting["bin_ops"], 
                                total_variables=list(eq_setting["total_variables"]),  
                                total_coefficients=list(eq_setting["total_coefficients"]),
                                rewrite_functions=list(eq_setting["rewrite_functions"]),
                                bfgs=bfgs,
                                beam_size=cfg.inference.beam_size #This parameter is a tradeoff between accuracy and fitting time
                                )

        weights_path = os.path.join(script_dir, "../tpsr/TPSR/nesymres/weights/10M.ckpt")

        ## Load architecture, set into eval mode, and pass the config parameters
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
