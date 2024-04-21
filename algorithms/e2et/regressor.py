import re
import torch
import numpy as np
import sympy as sp
import os, sys

# folder inside the git repo, not the repo itself!!
from .symbolicregression.model import SymbolicTransformerRegressor

import requests
from IPython.display import display


# Loading pre-trained model ----------------------------------------------------

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the model path relative to the script directory
model_path = os.path.join(script_dir, "model1.pt")

model = None
try:
    if not os.path.isfile(model_path): 
        url = "https://dl.fbaipublicfiles.com/symbolicregression/model1.pt"
        r = requests.get(url, allow_redirects=True)
        open(model_path, 'wb').write(r.content)
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


# Creating the model (symbolicregression is their package)
est = SymbolicTransformerRegressor(
    model=model,
    max_input_points=200,
    n_trees_to_refine=100,
    rescale=True
)


def model(est, X=None):
        
    replace_ops = {"add": "+", "mul": "*", "sub": "-", "pow": "**", "inv": "1/"}

    model_str = est.retrieve_tree(with_infos=True)["relabed_predicted_tree"].infix()

    for op,replace_op in replace_ops.items():
        model_str = model_str.replace(op,replace_op)
        
    return model_str