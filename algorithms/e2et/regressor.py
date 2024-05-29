import re
import torch
import numpy as np
import sympy as sp
import os, sys

from sklearn import feature_selection 
from sklearn.base import BaseEstimator, RegressorMixin

# from algorithms/e2et
sys.path.append('./')
sys.path.append('./symbolicregression')

# from experiments/methods
# sys.path.append('../algorithms/e2et/')
# sys.path.append('../algorithms/e2et/symbolicregression')

# folder inside the git repo, not the repo itself
from symbolicregression.model import SymbolicTransformerRegressor

from IPython.display import display


# Loading pre-trained model ----------------------------------------------------

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the model path relative to the script directory
model_path = os.path.join(script_dir, "model1.pt")

model = None
try:
    if not os.path.isfile(model_path): 
        raise(f"could not find model file in {model_path}")
    if not torch.cuda.is_available():
        model = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model = torch.load(model_path)
        model = model.cuda()
    print(model.device)
    print("Model successfully loaded!")
except Exception as e:
    print("ERROR: model not loaded! path was: {}".format(model_path))
    print(e)    


def get_top_k_features(X, y, k=10):
    if y.ndim==2:
        y=y[:,0]
    if X.shape[1]<=k:
        return [i for i in range(X.shape[1])]
    else:
        kbest = feature_selection.SelectKBest(feature_selection.r_regression, k=k)
        kbest.fit(X, y)
        scores = kbest.scores_
        top_features = np.argsort(-np.abs(scores))
        print("keeping only the top-{} features. Order was {}".format(k, top_features))
        return list(top_features[:k])


class E2ERegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        
        self.random_state = random_state

    def fit(self, X, y):
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)

        self.dstr = SymbolicTransformerRegressor(
            model=model,
            max_input_points=200,
            n_trees_to_refine=10,
            max_number_bags=10,
            rescale=True
        )

        x_to_fit = [X]
        y_to_fit = [y]

        self.dstr.fit(x_to_fit, y_to_fit, verbose=True)
    
        return self


    def predict(self, X):
        self.dstr.predict([X])


est = E2ERegressor()


def model(est, X=None):
        
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    model_str = est.dstr.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
        
    return model_str