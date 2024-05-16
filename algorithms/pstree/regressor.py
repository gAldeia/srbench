from pstree.cluster_gp_sklearn import PSTreeRegressor, GPRegressor
from sklearn.tree import DecisionTreeRegressor


est = PSTreeRegressor(
    regr_class=GPRegressor, tree_class=DecisionTreeRegressor,
    height_limit=6, n_pop=25, n_gen=100,
    basic_primitive='optimal', size_objective=True)


def model(est, X=None):
    return est.model()