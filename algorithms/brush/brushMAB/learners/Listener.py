#Can only be updated, but cannot make any choice. Used to keep track of events only
# (to have access to plot functions)

from .Base_Learner import Base_Learner

class Listener(Base_Learner):
    def __init__(self, num_bandits, pb=None, **kwargs):
        super(Listener, self).__init__()
        self.num_bandits = num_bandits

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen']}

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
        

    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)
        
        return self