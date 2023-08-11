"""Utility functions used in deep symbolic optimization."""

import multiprocessing
import collections
import copy
import functools
import numpy as np
import importlib
import random
import re
import os
import pandas as pd
from time import time
from typing import Callable

import sympy.parsing.sympy_parser as sympy_parser
import sympy


class Timer():
    """
    Utility class to help with walltime-based early stopping.

    It tracks records (e.g. a single RNN iteration, GP generation)
    and suggests whether to stop early to avoid exceeding walltime.
    """

    def __init__(self, max_time=None):
        self.start_time = time()
        self.records = []
        self.lap_time = None
        max_time = np.inf if max_time is None else max_time
        self.end_time = self.start_time + max_time

    def lap(self):
        current_time = time()
        if self.lap_time is not None:
            self.records.append(current_time - self.lap_time)
            print(multiprocessing.current_process(), "Lap:", self.records[-1])

        # Reset the lap time
        self.lap_time = current_time

    def clear_records(self):
        self.records = []

    def get_time_remaining(self):
        return self.end_time - time()

    def get_elapsed_time(self):
        return time() - self.start_time

    def stop_early(self, buffer=120., safety_factor=2.0):
        """
        Determine whether to stop early, based on worst-case records.
        This returns True if the amount of remaining time is less than
        buffer + (worst-case record) * safety_factor.
        """
        stop = self.get_time_remaining() < buffer + max(self.records) * safety_factor
        if stop:
            print(multiprocessing.current_process(), "\nTimed out! Stopping at {}, leaving {} remaining.".format(
                self.get_elapsed_time(), self.get_time_remaining()
            ))
            print("Longest record:", max(self.records))
        return stop


def preserve_global_rng_state(f: Callable):
    """
    Decorator that saves the internal state of the global random number
    generator before call to function and sets it back to that state
    after the call

    Parameters
    ----------
    f : Callable
        Function to decorate

    Returns
    _______
    Callable
        Decorated function that saves global random state and resets to it after
    """
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        rng_state = random.getstate()
        result = f(*args, **kwargs)
        random.setstate(rng_state)
        return result
    return decorated


# We wrap the sympy functions with preserve_global_rng_state
# as the sympy routines appear to non-deterministically
# re-seed the global random generator which can influence GP results.
# This problem seems to be resolved in sympy in commit
# https://github.com/sympy/sympy/pull/22433
# These functions should be used instead of the sympy functions directly
pretty = preserve_global_rng_state(sympy.pretty)
parse_expr = preserve_global_rng_state(sympy_parser.parse_expr)


def is_float(s):
    """Determine whether the input variable can be cast to float."""

    try:
        float(s)
        return True
    except ValueError:
        return False


# Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


def weighted_quantile(values, weights, q):
    """
    Computes the weighted quantile, equivalent to the exact quantile of the
    empirical distribution.

    Given ordered samples x_1 <= ... <= x_n, with corresponding weights w_1,
    ..., w_n, where sum_i(w_i) = 1.0, the weighted quantile is the minimum x_i
    for which the cumulative sum up to x_i is greater than or equal to 1.

    Quantile = min{ x_i | x_1 + ... + x_i >= q }
    """

    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    cum_sorted_weights = np.cumsum(sorted_weights)
    i_quantile = np.argmax(cum_sorted_weights >= q)
    quantile = sorted_values[i_quantile]

    # NOTE: This implementation is equivalent to (but much faster than) the
    # following:
    # from scipy import stats
    # empirical_dist = stats.rv_discrete(name='empirical_dist', values=(values, weights))
    # quantile = empirical_dist.ppf(q)

    return quantile


# Entropy computation in batch
def empirical_entropy(labels):

    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value,counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return np.array(ent, dtype=np.float32)


def get_duration(start_time):
    return get_human_readable_time(time() - start_time)


def get_human_readable_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "{:02d}:{:02d}:{:02d}:{:05.2f}".format(int(d), int(h), int(m), s)


def safe_merge_dicts(base_dict, update_dict):
    """Merges two dictionaries without changing the source dictionaries.

    Parameters
    ----------
        base_dict : dict
            Source dictionary with initial values.
        update_dict : dict
            Dictionary with changed values to update the base dictionary.

    Returns
    -------
        new_dict : dict
            Dictionary containing values from the merged dictionaries.
    """
    if base_dict is None:
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, collections.Mapping):
            base_dict[key] = safe_merge_dicts(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict


def safe_update_summary(csv_path, new_data):
    """Updates a summary csv file with new rows. Adds new columns
    in existing data if necessary. New rows are distinguished by
    the run seed.

    Parameters
    ----------
        csv_path : str
            String with the path to the csv file.
        new_data : dict
            Dictionary containing values to be saved in the csv file.

    Returns
    -------
        bool
            Boolean value to indicate if saving the data to file worked.
    """
    try:
        new_data_pd = pd.DataFrame(new_data, index=[0])
        new_data_pd.set_index('seed', inplace=True)
        if os.path.isfile(csv_path):
            old_data_pd = pd.read_csv(csv_path)
            old_data_pd.set_index('seed', inplace=True)
            merged_df = pd.concat([old_data_pd, new_data_pd], axis=0, ignore_index=False)
            merged_df.to_csv(csv_path, header=True, mode='w+', index=True)
        else:
            new_data_pd.to_csv(csv_path, header=True, mode='w+', index=True)
        return True
    except:
        return False


def import_custom_source(import_source):
    """
    Provides a way to import custom modules. The return will be a reference to the desired source
    Parameters
    ----------
        import_source : import path
            Source to import from, for most purposes: <module_name>:<class or function name>

    Returns
    -------
        mod : ref
            reference to the imported module
    """

    # Partially validates if the import_source is in correct format
    regex = '[\w._]+:[\w._]+' #lib_name:class_name
    m = re.match(pattern=regex, string=import_source)
    # Partial matches mean that the import will fail
    assert m is not None and m.end() == len(import_source), "*** Failed to import malformed source string: "+import_source

    source, type = import_source.split(':')

    # Dynamically imports the configured source
    mod = importlib.import_module(source)
    func = getattr(mod, type)

    return func

def pad_action_obs_priors(actions, obs, priors, pad_length):
    """
    Will pad action, obs, priors with zeros.  
    
    Parameters
    ----------
        actions : np array
            Standard actions array of tokens
        obs : np array
            Standard observations array
        priors : np array
            Standard priors array
        pdd_length : int

    Returns
    -------
        actions : np array
            Standard actions array of tokens padded with zeros at the end columns
        obs : np array
            Standard observations array padded with zeros at the end columns
        priors : np array
            Standard priors array padded with zeros at the end columns
    """
    assert isinstance(pad_length,int)
    assert pad_length >= 0
    
    actions = np.pad(actions, ((0,0),(0,pad_length)), 'constant', constant_values=((0,0),(0,0)))
    obs = [ np.pad(o, ((0,0),(0,pad_length)), 'constant', constant_values=((0,0),(0,0))) for o in obs ]
    priors = np.pad(priors, ((0,0),(0,pad_length),(0,0)), 'constant', constant_values=((0,0),(0,0),(0,0)))

    return actions, obs, priors



