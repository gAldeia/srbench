from .Base_Learner import Base_Learner
import numpy as np

class Dummy_Learner(Base_Learner):
    def __init__(self, num_bandits, pb=None, **kwargs):
        super(Dummy_Learner, self).__init__()

        self.rng = np.random.default_rng()
        self.num_bandits = num_bandits

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] +
                          [f'_probabilities {i}' for i in range(num_bandits)] } 

        # Using same information (HARDCODED) than default args
        if pb is None:
            self._probabilities = np.ones(num_bandits)/num_bandits
        else:
            assert len(pb)==num_bandits and sum(pb)==1.0, 'If you want to specify the probabilities, they must have same length than number of bandits, and need to sum up to 1.0'
            self._probabilities = pb

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
        """Tries to mimic how original brush samples: an mutation is chosen
        uniformly between options, except if the expression is already at
        maximum or minimum size/depth
        """
        
        # probability estimates from the beta distribution
        arm_idx = self.rng.choice(self.num_bandits, p=self._probabilities)
        
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

        # Never performs a dinamic update in its inner parameters 
        # (since it got no parameter) 
        self.pull_history['update'].append( 0 )
        
        # Now that we finished updating the values we save them to the logs
        for i in range(self.num_bandits):
            self.pull_history[f'_probabilities {i}'].append( self._probabilities[i] )


    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)
        
        return self
    