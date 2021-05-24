import sys
import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
import warnings
import time
from tempfile import mkdtemp
from shutil import rmtree
from joblib import Memory
from read_file import read_file
import pdb
import numpy as np
import json
import os
import inspect
from utils import jsonify 
from symbolic_utils import (clean_pred_model,get_sym_model,round_floats,
                            complexity, rewrite_AIFeynman_model_size)
from sympy import simplify

def save(r,save_file):
    print(json.dumps(r, indent=4))
    print('saving...')
    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(r), out, indent=4)


def fix_aifeynman_model_size(dataset, results_path, random_state, est_name,  
                   target_noise=0.0, feature_noise=0.0):

    print(40*'=','Assessing '+est_name+' model for ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)

    #################################################
    # load json file
    #################################################
    dataset_name = dataset.split('/')[-1][:-7]

    save_file = (results_path + '/' + dataset_name + '_' + est_name + '_' 
                 + str(random_state))
    if target_noise > 0:
        save_file += '_target-noise'+str(target_noise)
    if feature_noise > 0:
        save_file += '_feature-noise'+str(feature_noise)

    print('looking for:',save_file+'.json')

    if os.path.exists(save_file+'.json'):
        r = json.load(open(save_file+'.json', 'r'))
    else:
        raise FileNotFoundError(save_file+'.json not found')

    raw_model = r['symbolic_model']

    if 'AIFeynman' in est_name:
        # correct model size
        model_size = rewrite_AIFeynman_model_size(raw_model)
        print('changing model size from {} to {}'.format(
              model_size,r['model_size']))
        r['model_size'] = model_size
    else:
        raise ValueError('est_name not AIFeynman (it is %s)' % est_name)

    save(r, save_file)

    print('done.')

################################################################################
# main entry point
################################################################################
import argparse
import importlib

if __name__ == '__main__':

    # parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate a method on a dataset.", add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='ALG',default=None,type=str, 
            help='Name of estimator (with matching file in methods/)')
    parser.add_argument('-results_path', action='store', dest='RDIR',
                        default='results_test', type=str, 
                        help='Name of save file')
    parser.add_argument('-seed', action='store', dest='RANDOM_STATE',
                        default=42, type=int, help='Seed / trial')
    parser.add_argument('-test',action='store_true', dest='TEST', 
                       help='Used for testing a minimal version')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-sym_data',action='store_true', dest='SYM_DATA', 
                       help='Use symbolic dataset settings')

    args = parser.parse_args()

    print(args.__dict__)

    fix_aifeynman_model_size(args.INPUT_FILE, args.RDIR, args.RANDOM_STATE, args.ALG,
                   target_noise=args.Y_NOISE, feature_noise=args.X_NOISE
                   )
