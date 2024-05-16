import sys
import numpy as np
import pandas as pd

# from algorithms/tpsr
sys.path.append('./')
sys.path.append('./deep-symbolic-optimization-3.0.0/')

# from experiments/methods
# sys.path.append('../algorithms/deep-symbolic-optimization-3.0.0/')
# sys.path.append('../algorithms/deep-symbolic-optimization-3.0.0/dso')

from dso import DeepSymbolicOptimizer

X = np.array([[1,1],[2,2],[3,3]])
y = np.array([1, 4, 9])

# # Create and train the model
model = DeepSymbolicOptimizer("./blackbox_config.json")

model.config["task"]["dataset"] = (X, y)
# model.config["task"]["dataset"] = '../../datasets/pmlb/datasets/192_vineyard/192_vineyard.tsv.gz'

# Turn off file saving (otherwise it will try to serialize X and y)
model.config["experiment"]["logdir"] = None

# model.set_config(model.config)

train_result = model.train()
program_ = train_result["program"]


print("done")
