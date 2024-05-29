# Describes the basic interface for the learners
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rc("axes.spines", top=False, right=False)

class Base_Learner:
    # Learners will use parameters's default values inside brush Hacked.
    # Defalt values in __init__ function of learners should be compatible
    # with brush experiments. each main function in the learner's files should
    # work with these default values as well.
    def __init__(self, num_objectives=2, weights=(-1.0, -1.0), **kwargs):
        self.num_objectives = num_objectives
        self.weights = weights

    @property
    def probabilities(self):
        raise NotImplementedError()
    
    
    @probabilities.setter
    def probabilities(self, new_probabilities):
        raise NotImplementedError()


    def choose_arm(self, **context):
        raise NotImplementedError()
    
    # Methods to calculate the pareto dominance, used to define the reward
    def _epsi_less(self, a, b, e):
        return a < b and not self._epsi_eq(a, b, e)
    
    def _epsi_eq(self, a, b, e):
        return np.abs(a-b) <= e
    
    def _epsi_lesseq(self, a, b, e):
        return self._epsi_less(a, b, e) or self._epsi_eq(a, b, e)

    def _epsi_dominates(self, a, b, eps=None):

        if eps is None:
            eps = [1e-5 for _ in zip(a, b)]

        # Returns true if a DOMINATES b. We consider dominance as a minimization problem by default
        # a needs to be <= for all values in b, and strictly < for at least one value.
        # The weights must be +1 for maximization problems and -1 for minimization.
        return all(self._epsi_lesseq(ai, bi, ei) for ai, bi, ei in zip(a, b, eps)) \
        and    any(self._epsi_less(ai, bi, ei)   for ai, bi, ei in zip(a, b, eps))
    
    def _calc_reward(self, delta_costs, eps=None):
        delta_costs = np.array(delta_costs)
        # `delta_costs` will be a numpy array with multiple values (deltas for
        # each objective function). Each learner needs to figure out how it will
        # handle the values.

        # Example how to decide if the reward should be positive
        # At least one thing got better and the others didn't got worse
        # Rewards are always expected to be an array (even when it is one value)
        reward = 1.0 if (delta_costs[0]>0) else 0

        # dominance based reward
        if eps is None:
            eps = [1e-5 for _ in delta_costs] # one for each objective

        reward = 0.0 # in case none of if statements below work, this means offspring is dominated by the parent
        # if reward=1, then self._epsi_dominates(self.weights*delta_costs, np.zeros_like(delta_costs), eps, 
        # and A is non dominated by B (improved at least one objective)

        # proportional to number of objectives where offspring is not dominated
        # self._epsi_dominates(np.zeros_like(delta_costs), self.weights*delta_costs):# b dominates a (improved all objectives)
        num_improvements = sum(
            [1.0 if self._epsi_less(0, ai, ei) else 0.0
             for ai, ei in zip(self.weights*delta_costs, eps)])

        reward = num_improvements/len(self.weights)
        
        # Dominates
        # return 1.0 if reward==1.0 else 0.0
    
        # non-dominance
        return 1.0 if reward>0 else 0.0
    

    def log(self, arm_idx, delta_costs, success, **context):
        # Should be called inside update, or when a learner fails but still
        # need to report what happened.

        # If the log needs more information, then it should be accessible
        # via an private attribute (so the call for log is always the same)
        raise NotImplementedError()


    def update(self, arm_idx, delta_costs, success=True, **context):
        reward = self._calc_reward(delta_costs)

        # Success indicates if Brush could successfully perform the mutation, or
        # if it returned optional (which would be a failure). Optional can happen
        # for several reasons (no spot to change, no node to replace, expression
        # ended up being to big (exceeding max_size or max_depth).) In the future
        # we want to be able to learn weights for the spots and the nodes to replace,
        # so it is nice to have this information

        self.log(arm_idx, delta_costs, success, reward, **context)

        raise NotImplementedError()
    

    def calculate_statistics(self, arm_labels=[]):
    # getting the labels to use in plots
        if len(arm_labels) != self.num_bandits:
            arm_labels = [f'arm {i}' for i in range(self.num_bandits)]

        learner_log = pd.DataFrame(self.pull_history).set_index('t')
        
        # TODO: rename it to total_non-empty_rewards (or something similar), because reward can be 0.5
        total_rewards = {i:0 for i in range(self.num_bandits)}
        total_rewards.update(learner_log[learner_log['reward']>0.0].groupby('arm idx')['reward'].size().to_dict())

        total_half_rewards = {i:0 for i in range(self.num_bandits)}
        total_half_rewards.update(learner_log[(learner_log['reward']<1.0) & (learner_log['reward']>0.0)
                                              ].groupby('arm idx')['reward'].size().to_dict())

        total_success = {i:0 for i in range(self.num_bandits)}
        total_success.update(learner_log.groupby('arm idx')['success'].sum().to_dict())

        # avoiding have a different number of values in dict when one or more arms are not used
        total_pulls = {i:0 for i in range(self.num_bandits)}
        total_pulls.update(learner_log['arm idx'].value_counts().to_dict())

        data_total_pulls        = np.array([total_pulls[k] for k in sorted(total_pulls)])
        data_total_rewards      = np.array([total_rewards[k] for k in sorted(total_rewards)])
        data_total_half_rewards = np.array([total_half_rewards[k] for k in sorted(total_half_rewards)])
        data_total_success      = np.array([total_success[k] for k in sorted(total_success)])
        data_total_failures     = data_total_pulls-data_total_success

        statistics = pd.DataFrame.from_dict({
            'arm'         : arm_labels,
            'totpulls'    : data_total_pulls,
            'success'     : data_total_success,
            'pulls%'      : np.nan_to_num( (data_total_pulls/data_total_pulls.sum()).round(2) ),
            'success%'    : np.nan_to_num( (data_total_success/data_total_pulls).round(2) ),
            '+reward'     : data_total_rewards,
            'part. reward': data_total_half_rewards,
            'win rate'    : np.nan_to_num( (data_total_success/np.minimum(1, data_total_failures)).round(2) ), # if the arm was not pulled once, it ends dividing by zero. nan_to_sum is to present some interpretable value instead of NaN
        })

        return statistics