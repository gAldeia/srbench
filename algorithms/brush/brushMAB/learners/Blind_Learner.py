from .Base_Learner import Base_Learner
import numpy as np

class Blind_Learner(Base_Learner):
    def __init__(self, num_bandits, update_strength=0.005, **kwargs):
        super(Blind_Learner, self).__init__()

        self.rng = np.random.default_rng()
        self.num_bandits = num_bandits
        self.update_strength = update_strength

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error', 'gen'] +
                          [f'_probabilities {i}' for i in range(num_bandits)] } 

        # Starts with a uniform distribution, and update weights by a small amount
        # sampled by an uniform distribution
        self._probabilities = np.ones(num_bandits)/num_bandits

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

        # Normal distribution vector to update the probabilities
        pb_change = np.random.normal(size=self.num_bandits)*self.update_strength
        pb = np.maximum(self.probabilities + pb_change, 0)
        pb_sum = np.sum(pb)
        self.probabilities = pb/pb_sum

    def update(self, arm_idx, delta_costs, success=1, **context):
        self.log(arm_idx, delta_costs, success, **context)
        
        return self
    

def main(show=False):
    from learners.Base_Learner import Bandits
    import numpy as np
    
    for probs, descr, expec in [
        (
            np.array([ 1.0,  1.0, 1.0,  1.0]),
            'All bandits with same probs',
            'similar amount of pulls for each arm'
        ),
        (
            np.array([-1.0,  0.2, 0.0,  1.0]),
            'One bandit with higher prob',
            'more pulls for first arm, less pulls for last'
        ),
        (
            np.array([-0.2, -1.0, 0.0, -1.0]),
            'Two bandits with higher probs',
            '2nd approx 4th > 1st > 3rd'
        ),
    ]:
        print("-------------------------- optimizing --------------------------")

        learner = Blind_Learner(4)
        
        bandits = Bandits(probs)
        learner.sanity_check(bandits)

        initial = learner.calculate_statistics()
        print("initial statistics")
        print(initial)

        # learner.plot_learner_history(show=True)
        # learner.plot_statistics(show=True)

        print(f"(it was expected: {expec})")
        print("Changing to uniform reward distribution to see if it dynamically responds")

        bandits = Bandits(np.array([ 1.0,  1.0, 1.0,  1.0]))
        learner.sanity_check(bandits)
        final = learner.calculate_statistics()
        print("statistics after change")
        print(final)

        print("Difference in statistics")
        print(final.set_index('arm').sub(initial.set_index('arm')).reset_index())

        # learner.plot_learner_history(show=True)
        # learner.plot_statistics(show=True)
        
        print(f"(the difference should have similar amount of pulls for each arm)")


if __name__ == "__main__":
    main()