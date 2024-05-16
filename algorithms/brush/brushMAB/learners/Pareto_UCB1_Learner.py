# Implementation of https://ieeexplore.ieee.org/document/6707036

from .Base_Learner import Base_Learner
import numpy as np

class Pareto_UCB1_Learner(Base_Learner): # original MAB (why not?)
    def __init__(self, num_bandits, num_objectives=2, weights=(-1.0, -1.0), epsi=1e-5, **kwargs):
        super(Pareto_UCB1_Learner, self).__init__()

        # This learner samples an arm based on the UCB score for each arm.
        # There's always uncertainty in which arm is the best but, if optimistic
        # is true, then it chooses the arm with maximum UCB score (this is the
        # UCB1 algorithm).

        self.rng            =  np.random.default_rng()
        self.num_bandits    = num_bandits
        self.num_objectives = num_objectives
        self.weights        = np.array(weights)
        self.epsi           = epsi

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] + 
                          [f'_probabilities {i}'    for i in range(num_bandits)] +
                          [f'_num_pulls {i}'        for i in range(num_bandits)] }
        
        # each arm has an avg reward for each obj. Here, we'll consider the first objective as dominant
        self.pull_history.update({
            f'_avg_rewards {i}':[] for i in range(num_bandits)
        })                          
        
        # This is the probability that should be used to update brush probs
        self._probabilities = np.ones(num_bandits)/num_bandits
        
        # internally, its going to be a matrix. We use lists in pull_history to make it compatible with plot funcitons.
        # Each row is an arm, each column is an objective. Thus, _avg_rewards \in (K x D)
        self._avg_rewards = np.zeros( (num_bandits, num_objectives) ) # positive values are better (even for minimization problems, due to the weights specification)
        self._num_pulls   = np.ones(num_bandits)

    def _find_pareto_set(self, candidates):
        # Returns the indexes of tuples in `candidates` that are in the pareto set
        pareto_set = []
        for i, current_tuple in enumerate(candidates):
            is_dominated = False
            for j, other_tuple in enumerate(candidates):
                if i != j:
                    if self._epsi_dominates(other_tuple, current_tuple):
                        is_dominated = True
                        break

            # If the current tuple is not dominated by any other, then it is in the pareto set
            if not is_dominated:
                pareto_set.append(i)

        return np.array(pareto_set).astype(int)

    @property
    def probabilities(self):
        # How to transform our UCB1 scores into node probabilities?
        return self._probabilities
    
    @probabilities.setter
    def probabilities(self, new_probabilities):
        if len(self._probabilities)==len(new_probabilities):
            self._probabilities = new_probabilities
        else:
            print(f"New probabilities must have size {self.num_bandits}")

    def choose_arm(self, **context):
        # I think I should really improve overall code redability... especially the equations

        # self._avg_rewards already takes into account the weights (so every
        # objective is seen as a minimization, which is expected by find_pareto_set and _is_dominated)
        #astar = self._find_pareto_set(self._avg_rewards)
        astar = np.array([i for i in range(self.num_bandits)])
        confidence_intervals = np.sqrt( 2*np.log1p(sum(self._num_pulls)*np.sqrt(np.sqrt(self.num_objectives*len(astar))))/(self._num_pulls) ) # log1p to avoid numerical errors

        # Multiply by -1 just because our find_pareto_set uses the _epsi_dominated, which
        # considers smaller values better (and reward is always positive, and bigger values
        # are better)
        aprime = self._find_pareto_set(-1.0 * (self._avg_rewards[astar, :] + (confidence_intervals[astar])[np.newaxis].T*np.ones( (1, self.num_objectives)) ))

        # OBS: i changed the confidence interval to be like the original UCB1 because the proposed
        # version in the paper doesn't work for single objective (which I think it should). this change,
        # although, lacks a mathematical verification (but empirically has shown better results)

        assert len(aprime)>0, "failed to find the pareto set"

        # aprime is an index of the selected arm in astar, which is also a subset of all the arms
        # print(astar)
        # print(aprime)
        # print(np.random.choice(astar[aprime]))

        return self.rng.choice(astar[aprime])

    def log(self, arm_idx, delta_costs, success, **context):
        reward = self._calc_reward(delta_costs)

        # Here we expect that the reward was already scaled to be in the 
        # interval [0, 1] (in the original paper, they sugest using a scaling
        # factor as an hyperparameter).
        self.pull_history['t'].append( len(self.pull_history['t']) )
        self.pull_history['arm idx'].append( arm_idx )
        self.pull_history['reward'].append( reward )
        self.pull_history['success'].append( success )
        self.pull_history['delta error'].append( self.weights*delta_costs )
        self.pull_history['gen'].append( context['gen'] )

        # Since this is not a dynamic implementation, this will never change
        self.pull_history['update'].append( 0 )

        self._probabilities = [1.0 if arm in self._find_pareto_set(self._avg_rewards) else 0.0
                                   for arm in range(self.num_bandits)]

        # Update history
        for i in range(self.num_bandits):
            self.pull_history[f'_probabilities {i}'].append( self.probabilities[i] )
            self.pull_history[f'_num_pulls {i}'].append( self._num_pulls[i] )
            self.pull_history[f'_avg_rewards {i}'].append( self._avg_rewards[i][0] )


    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)

        # Reward vector is used to update the counters for each objective.
        # This is done to avoid having negative values affecting avg rewards (which makes the learner behaves
        # like dummy learner)
        reward_vector = np.array([1.0 if self._epsi_less(0, ai, ei) else 0.0
                                  for ai, ei in zip(self.weights*delta_costs, [1e-5, 1e-5])])
        
        # Updating counters, for every objective (second array dimension)
        self._num_pulls[arm_idx]   = self._num_pulls[arm_idx] +1
        self._avg_rewards[arm_idx, :] = self._avg_rewards[arm_idx, :] + ((reward_vector - self._avg_rewards[arm_idx, :])/sum(self._num_pulls)) # here we are updating a mean. need to adjust contribution of new value 

        return self
    