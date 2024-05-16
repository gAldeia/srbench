from .Base_Learner import Base_Learner
import numpy as np

class D_UCB1_Learner(Base_Learner):
    def __init__(self, num_bandits, delta=2, lmbda=100, **kwargs):
        super(D_UCB1_Learner, self).__init__()
        
        self.num_bandits = num_bandits

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] + 
                          [f'_UCB1s {i}'            for i in range(num_bandits)] + 
                          [f'_avg_rewards {i}'      for i in range(num_bandits)] +
                          [f'_probabilities {i}'    for i in range(num_bandits)] +
                          [f'_num_pulls {i}'        for i in range(num_bandits)] +
                          [f'_max_deviations {i}'   for i in range(num_bandits)] +
                          [f'_avg_deviations {i}'   for i in range(num_bandits)] }
        
        # This is the probability that should be used to update brush probs
        self._probabilities = np.ones(num_bandits)/num_bandits

        self._UCB1s = np.zeros(self.num_bandits)
        self.delta = delta # how to define these values???
        self.lmbda = lmbda

        self._reset_indicators() # Creating the indicators 

    def _reset_indicators(self):
        self._avg_rewards    = np.zeros(self.num_bandits)
        self._num_pulls      = np.zeros(self.num_bandits)
        self._avg_deviations = np.zeros(self.num_bandits)
        self._max_deviations = np.zeros(self.num_bandits)

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
        #print(self._UCB1s)
        #print(self._num_pulls)
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

        UCB1s = self._calculate_UCB1s()
        self._probabilities = UCB1s

        # Update history
        for i, UCB1 in enumerate(UCB1s):
            self.pull_history[f'_UCB1s {i}'].append( UCB1 )
            self.pull_history[f'_probabilities {i}'].append( self.probabilities[i] )
            self.pull_history[f'_avg_rewards {i}'].append( self._avg_rewards[i] )
            self.pull_history[f'_num_pulls {i}'].append( self._num_pulls[i] )
            self.pull_history[f'_max_deviations {i}'].append( self._max_deviations[i] )
            self.pull_history[f'_avg_deviations {i}'].append( self._avg_deviations[i] )

        if (self._max_deviations[arm_idx] - self._avg_deviations[arm_idx] > self.lmbda):
            self._reset_indicators()
            self.pull_history['update'].append( 1 )
        else:
            self.pull_history['update'].append( 0 )


    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)

        reward = self._calc_reward(delta_costs)

        # print("updating")
        # Updating counters
        self._num_pulls[arm_idx]    = self._num_pulls[arm_idx] +1
        self._avg_rewards[arm_idx]  = self._avg_rewards[arm_idx] + ((reward - self._avg_rewards[arm_idx])/self._num_pulls[arm_idx])
        self._avg_deviations[arm_idx] = self._avg_deviations[arm_idx] + (self._avg_rewards[arm_idx] - reward + self.delta)    
        self._max_deviations[arm_idx] = np.maximum(self._max_deviations[arm_idx], self._avg_deviations[arm_idx])

        return self