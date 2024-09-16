import sys
import itertools
import pandas as pd
from sklearn.base import clone
from sklearn.experimental import enable_halving_search_cv # noqa
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  
from sklearn.model_selection import train_test_split
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
import sympy as sp
import inspect
from utils import jsonify
from symbolic_utils import get_sym_model

from metrics.evaluation import simplicity

import signal
class TimeOutException(Exception):
    pass

def alarm_handler(signum, frame):
    print(f"raising TimeOutException")
    raise TimeOutException

def set_env_vars(n_jobs):
    os.environ['OMP_NUM_THREADS'] = n_jobs 
    os.environ['OPENBLAS_NUM_THREADS'] = n_jobs 
    os.environ['MKL_NUM_THREADS'] = n_jobs


def _save_brush_evolution(estimator, name, dataset, random_state, rdir, repeat):
    """Save a file with evolution info. used in convergence analysis"""
    logbook = pd.DataFrame(columns=['gen', 'evals', 'best_size'] + 
                            [f"{stat} {partition} {objective}"
                             for stat in ['avg', 'med', 'std', 'min', 'max']
                             for partition in ['train', 'val']
                             for objective in estimator.objectives])
    
    for item in estimator.logbook_:
        logbook.loc[item['gen']] = (
            item['gen'], item['evals'], item['best_size'],
            *item['avg'], *item['med'], *item['std'], *item['min'], *item['max']
        )

    logbook_filename = rdir + '_'.join([dataset, 
                                        name, 
                                        str(repeat), 
                                        str(random_state),
                                        "evolution"]) + '.csv'
        
    pd.DataFrame.from_dict({col: logbook[col] for col in logbook.columns}
    ).to_csv(logbook_filename, index=False)

def evaluate_model(
    dataset, 
    results_path,
    random_state,
    est_name,
    est,
    model,
    test=False,
    sym_data=False,
    target_noise=0.0, 
    feature_noise=0.0, 
    ##########
    # valid options for eval_kwargs
    ##########
    test_params={},
    max_train_samples=0,
    scale_x=True,
    scale_y=True,
    pre_train=None,
    use_dataframe=True
):

    print(40*'=','Evaluating '+est_name+' on ',dataset,40*'=',sep='\n')

    np.random.seed(random_state)
    if hasattr(est, 'random_state'):
        est.random_state = random_state

    ##################################################
    # setup data
    ##################################################

    if est_name == "e2et" or est_name == 'tpsr' or ('nesymres' in est_name) \
    or est_name == "dso":
        use_dataframe = False
    
    ##################################################
    # setup data
    ##################################################
    features, labels, feature_names =  read_file(
        dataset, 
        use_dataframe=use_dataframe
    )

    # if est_name == 'tpsr': # TODO: remove it
    #     if len(feature_names)>10: # TPSR works with at most 10 features
    #         raise Exception('WARNING: TPSR does not support more than 10 features. Skipping')

    print('feature_names:',feature_names)
    if sym_data:
        true_model = get_sym_model(dataset)

    # generate train/test split
    X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    train_size=0.75,
                                                    test_size=0.25,
                                                    random_state=random_state)

    # time limits
    MAXTIME = 7_200
    if len(y_train) > 1000:
        MAXTIME = 72_000

    print('max time:',MAXTIME)

    # if dataset is large, subsample the training set 
    if max_train_samples > 0 and len(y_train) > max_train_samples:
        print('subsampling training data from',len(X_train),
              'to',max_train_samples)
        sample_idx = np.random.choice(np.arange(len(X_train)),
                                      size=max_train_samples,
                                      replace=False)
        y_train = y_train[sample_idx]
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.iloc[sample_idx]
        else:
            X_train = X_train[sample_idx]

    # scale and normalize the data
    if scale_x:
        print('scaling X')
        sc_X = StandardScaler() 
        X_train_scaled = sc_X.fit_transform(X_train)
        X_test_scaled = sc_X.transform(X_test)
        if use_dataframe:
            X_train_scaled = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
            X_test_scaled = pd.DataFrame(X_test_scaled, 
                                          columns=feature_names)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test


    if scale_y:
        print('scaling y')
        sc_y = StandardScaler()
        y_train_scaled = sc_y.fit_transform(y_train.reshape(-1,1)).flatten()
    else:
        y_train_scaled = y_train


    ################################################## 
    # noise
    ################################################## 
    if target_noise > 0:
        print('adding',target_noise,'noise to target')
        y_train_scaled += np.random.normal(0, 
                    target_noise*np.sqrt(np.mean(np.square(y_train_scaled))),
                    size=len(y_train_scaled))
    # add noise to the features
    if feature_noise > 0:
        print('adding',target_noise,'noise to features')
        X_train_scaled = np.array([x 
            + np.random.normal(0, feature_noise*np.sqrt(np.mean(np.square(x))),
                               size=len(x))
                                   for x in X_train_scaled.T]).T

    ################################################## 
    # run any method-specific pre_train routines
    ################################################## 
    if pre_train:
        pre_train(est, X_train_scaled, y_train_scaled)

    # define a test mode using estimator test_params, if they exist
    if test and len(test_params) != 0:
        est.set_params(**test_params)

    ################################################## 
    # Fit models
    ################################################## 
    if not use_dataframe: 
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)

    print('X_train:',type(X_train_scaled),X_train_scaled.shape)
    print('y_train:',y_train_scaled.shape)
    print('training',est)
    t0t = time.time()
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(MAXTIME) # maximum time, defined above
    try:
        est.fit(X_train_scaled, y_train_scaled)
        
        if "brush" in est_name and "cpp" not in est_name: # saving log
            dataset_name = dataset.split('/')[-1].split('.')[0]
            
            # this can use a lot of space
            # _save_brush_evolution(est, est_name, dataset_name, random_state, results_path, 0)

            # Printing archive
            for ind_idx, ind in enumerate(est.archive_):
                print(ind_idx, ind.prg.get_model(), ind.fitness)
    
    except TimeOutException:
        print('WARNING: fitting timed out')

    time_time = time.time() - t0t
    print('Training time measure:', time_time)
    
    ##################################################
    # store results
    ##################################################
    dataset_name = dataset.split('/')[-1].split('.')[0]
    results = {
        'dataset':dataset_name,
        'algorithm':est_name,
        'params':jsonify(est.get_params()),
        'random_state':random_state,
        'time_time': time_time, 
    }
    if sym_data:
        results['true_model'] = true_model

    # get the final symbolic model as a string
    print('fitted est:',est)

    if 'X' in inspect.signature(model).parameters.keys():
        if not isinstance(X_train_scaled, pd.DataFrame):
            X_df = pd.DataFrame(X_train_scaled, 
                                          columns=feature_names)
        else:
            X_df = X_train_scaled
        results['symbolic_model'] = model(est, X_df)
    else:
        results['symbolic_model'] = model(est)
        
    print('symbolic model:',results['symbolic_model'])
    ##################################################
    # scores
    ##################################################

    for fold, target, X in  [ 
                             ['train', y_train, X_train_scaled], 
                             ['test', y_test, X_test_scaled]
                            ]:

        y_pred = np.asarray(est.predict(X)).reshape(-1,1)
        if scale_y:
            y_pred = sc_y.inverse_transform(y_pred)

        for score, scorer in [('mse',mean_squared_error),
                              ('mae',mean_absolute_error),
                              ('r2', r2_score)
                             ]:
            results[score + '_' + fold] = scorer(target, y_pred) 
    
    # simplicity
    results['simplicity'] = simplicity(results['symbolic_model'], feature_names)

    # model size for feat
    if 'feat' in est_name:
        def get_complexity(est):
            complexity = est.get_n_nodes() + 2*est.get_n_params() + 2*est.get_dim()
            return complexity
        results['model_size'] = get_complexity(est)

    if "brush" in est_name:
        if "cpp" in est_name:
            results['model_size'] = est.best_estimator_.fitness.size
        else: # older versions of brush 
            results['model_size'] = est.best_estimator_.size()

    if "tpsr" in est_name:
        results['model_size'] = len(est.tree_[0])
        
    if "nesymres" in est_name:
        def get_sympy_expr_size(expr):
            print(expr)
            tot_size = 1
            for arg in expr.args:
                tot_size += get_sympy_expr_size(arg)
            return tot_size
            
        expr = sp.parse_expr(est.model_eq_[0])
        results['model_size'] = get_sympy_expr_size(expr)

    if "dso" in est_name:
        results['model_size'] = len(est.program_.traversal)

    if "e2et" in est_name: 
        # print("retrieving tree", est.retrieve_tree(with_infos=True))

        model_tree = est.dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"]
        
        results['model_size'] = len(model_tree)


    results['target_noise']  = args.Y_NOISE
    results['feature_noise'] = args.X_NOISE

    ##################################################
    # write to file
    ##################################################
    print('results:')
    print(json.dumps(results,indent=4))
    print('---')

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    save_file = os.path.join(
        results_path,
        '_'.join([dataset_name, est_name, str(random_state)])
    )
    if args.Y_NOISE > 0:
        save_file += '_target-noise'+str(args.Y_NOISE)
    if args.X_NOISE > 0:
        save_file += '_feature-noise'+str(args.X_NOISE)
        
    print('save_file:',save_file)

    with open(save_file + '.json', 'w') as out:
        json.dump(jsonify(results), out, indent=4)

    return save_file + '.json'

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
    parser.add_argument('-n_jobs',action='store',  type=str, default='4',
                        help='number of cores available')
    parser.add_argument('-max_samples',action='store',  type=int, default=0,
                        help='number of training samples')
    parser.add_argument('-target_noise',action='store',dest='Y_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    parser.add_argument('-feature_noise',action='store',dest='X_NOISE',
                        default=0.0, type=float, help='Gaussian noise to add'
                        'to the target')
    
    parser.add_argument('--sym_data', action='store_false', dest='SYM_DATA', default=False)
    parser.add_argument('--scale_x', action='store_true', dest='SCALE_X', default=False) 
    parser.add_argument('--scale_y', action='store_true', dest='SCALE_Y', default=False)

    parser.add_argument('-skip_tuning',action='store_true', dest='SKIP_TUNE', 
                        default=False, help='Dont tune the estimator')

    args = parser.parse_args()
    set_env_vars(args.n_jobs)
    # import algorithm 
    print('import from','methods.'+args.ALG+'.regressor')
    algorithm = importlib.__import__('methods.'+args.ALG+'.regressor',
                                     globals(),
                                     locals(),
                                     ['*']
                                    )

    print('algorithm:',algorithm.est)

    # optional keyword arguments passed to evaluate
    eval_kwargs, test_params = {},{}
    if 'eval_kwargs' in dir(algorithm):
        eval_kwargs = algorithm.eval_kwargs

    if args.max_samples != 0:
        eval_kwargs['max_train_samples'] = args.max_samples

    eval_kwargs['sym_data'] = args.SYM_DATA
    eval_kwargs['scale_x'] = args.SCALE_X
    eval_kwargs['scale_y'] = args.SCALE_Y

    evaluate_model(args.INPUT_FILE,
                   args.RDIR,
                   args.RANDOM_STATE,
                   args.ALG,
                   algorithm.est,  
                   algorithm.model, 
                   test = args.TEST, 
                   **eval_kwargs
                  )
