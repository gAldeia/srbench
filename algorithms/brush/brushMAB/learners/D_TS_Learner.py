from .Base_Learner import Base_Learner
import numpy as np

class D_TS_Learner(Base_Learner):
    def __init__(self, num_bandits, C=5000, **kwargs):
        super(D_TS_Learner, self).__init__()
        
        self.rng         = np.random.default_rng()
        self.num_bandits = num_bandits

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] + 
                          [f'_alphas {i}'  for i in range(num_bandits)] + 
                          [f'_betas {i}'   for i in range(num_bandits)] + 
                          [f'_probabilities {i}' for i in range(num_bandits)] } 

        # This is the probability that should be used to update brush probs
        self._probabilities = np.ones(num_bandits)/num_bandits

        self._alphas = 2*np.ones(num_bandits) # Paper suggests starting with 2's
        self._betas  = 2*np.ones(num_bandits)
        self.C       = C # how to define this value???

    @property
    def probabilities(self):
        return self._probabilities
    
    @probabilities.setter
    def probabilities(self, new_probabilities):
        if len(self._probabilities)==len(new_probabilities):
            self._probabilities = new_probabilities
        else:
            print(f"New probabilities must have size {self.num_bandits}")

    def choose_arm(self, **context):
        """Uses the learned distributions to randomly choose an arm to pull. 
        
        Returns the index of the arm that was choosen based on the Beta
        probabilities of previous successes and fails.
        """
        
        # probability estimates from the beta distribution
        thetas = self.rng.beta(self._alphas, self._betas)

        arm_idx = np.argmax(thetas)
        
        return arm_idx
    
    def log(self, arm_idx, delta_costs, success, **context):
        reward = self._calc_reward(delta_costs)

        # There are informations about state. we'll save the pull history of
        # other stuff after updating their values
        self.pull_history['t'].append( len(self.pull_history['t']) )
        self.pull_history['arm idx'].append( arm_idx )
        self.pull_history['reward'].append( reward )
        self.pull_history['success'].append( success )
        self.pull_history['delta error'].append( self.weights*delta_costs )
        self.pull_history['gen'].append( context['gen'] )

        if self._alphas[arm_idx] + self._betas[arm_idx] < self.C:
            self.pull_history['update'].append( 0 )
        else:
            self.pull_history['update'].append( 1 )

        # Now that we finished updating the values we save them to the logs
        for i in range(self.num_bandits):
            self.pull_history[f'_alphas {i}'].append( self._alphas[i] )
            self.pull_history[f'_betas {i}'].append( self._betas[i] )
            self.pull_history[f'_probabilities {i}'].append( self._probabilities[i] )


    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)
        
        reward = self._calc_reward(delta_costs)
        
        if self._alphas[arm_idx] + self._betas[arm_idx] < self.C:
            # This is the pure thompson scheme
            self._alphas[arm_idx] = self._alphas[arm_idx]+reward
            self._betas[arm_idx]  = self._betas[arm_idx]+(1-reward)
        else:
            # This is the dynamic adjust
            self._alphas[arm_idx] = (self._alphas[arm_idx]+reward)*(self.C/(self.C+1))
            self._betas[arm_idx]  = (self._betas[arm_idx]+(1-reward))*(self.C/(self.C+1))

        # How to transform our Beta distributions into node probabilities?
        # onde idea is to return the expected value of this distribution as
        # the weight that will be given to each arm. In the case of our prior
        # (which is a beta distribution), the expected value is given by
        # 1 / (1 + beta/alpha)
        #self._probabilities = 1 / (1 + (self._betas/self._alphas))
        self._probabilities = (self._alphas-1)/(self._alphas+self._betas-2)

        return self