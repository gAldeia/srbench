# Good reference: https://webdocs.cs.ualberta.ca/~games/go/seminar/notes/2007/slides_ucb.pdf

from .Base_Learner import Base_Learner
import numpy as np

class UCB1_Learner(Base_Learner): # original MAB (why not?)
    def __init__(self, num_bandits, **kwargs):
        super(UCB1_Learner, self).__init__()
        
        # This learner samples an arm based on the UCB score for each arm.
        # There's always uncertainty in which arm is the best but, if optimistic
        # is true, then it chooses the arm with maximum UCB score (this is the
        # UCB1 algorithm).

        self.num_bandits = num_bandits

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] + 
                          [f'_UCB1s {i}'            for i in range(num_bandits)] + 
                          [f'_avg_rewards {i}'      for i in range(num_bandits)] +
                          [f'_probabilities {i}'    for i in range(num_bandits)] +
                          [f'_num_pulls {i}'        for i in range(num_bandits)] }
        
        # This is the probability that should be used to update brush probs
        self._probabilities = np.ones(num_bandits)/num_bandits

        self._UCB1s       = np.zeros(num_bandits)
        self._avg_rewards = np.zeros(num_bandits)
        self._num_pulls   = np.zeros(num_bandits)

    def _calculate_UCB1s(self):
        # We need that the reward is in [0, 1] (not avg_reward, as it seems to
        # render worse results). It looks like normalizing the rewards is a
        # problem: reward should be [0, 1], but not necessarely avg_rewards too
        rs = self._avg_rewards
        ns = self._num_pulls
        
        return rs + np.sqrt(2*np.log1p(sum(ns))/(ns+1))

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
        """Uses previous recordings of rewards to pick the arm that maximizes
        the UCB1 function. The choice is made in a deterministic way.
        """

        self._UCB1s = self._calculate_UCB1s()

        return np.nanargmax(self._UCB1s)
    
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

        UCB1s = self._calculate_UCB1s()
        self._probabilities = UCB1s

        # Update history
        for i, UCB1 in enumerate(UCB1s):
            self.pull_history[f'_UCB1s {i}'].append( UCB1 )
            self.pull_history[f'_probabilities {i}'].append( self.probabilities[i] )
            self.pull_history[f'_avg_rewards {i}'].append( self._avg_rewards[i] )
            self.pull_history[f'_num_pulls {i}'].append( self._num_pulls[i] )

        return self
    
    
    def update(self, arm_idx, delta_costs, success=1, **context):
        reward = self._calc_reward(delta_costs)
        
        self.log(arm_idx, delta_costs, success, **context)

        # Updating counters
        self._num_pulls[arm_idx]    = self._num_pulls[arm_idx] +1
        self._avg_rewards[arm_idx]  = self._avg_rewards[arm_idx] + ((reward - self._avg_rewards[arm_idx])/self._num_pulls[arm_idx])
