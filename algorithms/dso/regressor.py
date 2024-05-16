from dso import DeepSymbolicRegressor
from dso import DeepSymbolicOptimizer
import sys

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

# from algorithms/dso
sys.path.append('./')

# from experiments/methods
# sys.path.append('../algorithms/dso/deep-symbolic-optimization-3.0.0/dso')

# using their sklearn wrapper (does not support GP steps)
# est = DeepSymbolicRegressor()


class uDSRRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, random_state=42):
        
        self.random_state = random_state

    def fit(self, X, y):        
        # # Create and train the model
        model = DeepSymbolicOptimizer("./blackbox_config.json")

        model.config["task"]["dataset"] = (X, y)
        # model.config["task"]["dataset"] = '../../datasets/pmlb/datasets/192_vineyard/192_vineyard.tsv.gz'

        # Turn off file saving (otherwise it will try to serialize X and y)
        model.config["experiment"]["logdir"] = None

        model.set_config(model.config)

        train_result = model.train()
        self.program_ = train_result["program"]

        return self

    def predict(self, X):
        check_is_fitted(self, "program_")

        return self.program_.execute(X)

est = uDSRRegressor()


# Should work for both methods
def model(est, X=None):
    # clean_pred_model from assess_symbolic model will fix the string rep
    return str(est.program_.sympy_expr)

    
if __name__ == "__main__":
    import numpy as np  
    
    # Generate some data
    np.random.seed(0)
    X = np.random.random((10, 2))
    y = np.sin(X[:,0]) + X[:,1] ** 2

    # Fit the model
    est.fit(X, y) # Should solve in ~10 seconds

    # View the best expression
    print(est.program_.pretty())
    print(est.program_.sympy_expr)

    print(len(est.program_.traversal))

    # Make predictions
    est.predict(2 * X)
