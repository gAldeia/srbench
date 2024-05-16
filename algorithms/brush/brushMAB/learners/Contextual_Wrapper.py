from .Base_Learner import Base_Learner
import numpy as np

class Ball():
    def __init__(self, center, radius, Learner, Learner_kwargs, id):
        self.center = center
        self.radius  = radius

        self.counter = 0
        self.learner = Learner(**Learner_kwargs)

        self.id = id # just a number to identify it (and the order they were created)
        self.active = True

    def choose_arm(self, **context):
        return self.learner.choose_arm(**context)

    def update(self, arm_idx, delta_costs, success=1, **context):
        self.learner.update(arm_idx, delta_costs, success, **context)
        self.counter = self.counter + 1


class Context_Space():
    def __init__(self, *arrays, d=None):
        # our context set is the cartesian product of each interval in `arrays`

        if len(arrays) == 0:
            raise ValueError("At least one array required as input")
        
        if d == None: # Euclidean distance is the default, upper bounded by 1 (1 should be the diameter of the search space)
            # d = lambda x, xprime: np.minimum( # x and xprime must be of same size than n_dimensions
            #     1.0, np.linalg.norm(self._get_context_set(x) - self._get_context_set(xprime))
            # )
            
            n_dimensions = len(arrays)
            max_distance = ( #np.linalg.norm
                np.array([np.max(d) for d in arrays]) - 
                np.array([np.min(d) for d in arrays])
            )

            d = lambda x, xprime: np.minimum( # x and xprime must be of same size than n_dimensions (number of contexts)
                1.0, np.linalg.norm(
                    (self._get_context_set(x) - self._get_context_set(xprime))/ max_distance
                ) 
            )

        self.n_dimensions = len(arrays)
        self.dimensions   = arrays
        self.d            = d

    def _get_context_set(self, x): # takes informations and return the context partition indexes
        return np.array([np.searchsorted(d_i, x_i)
                         for (d_i, x_i) in zip(self.dimensions, x)])
    

class Contextual_Wrapper(Base_Learner):
    def __init__(self, num_bandits, Learner, Learner_kwargs={}, context_space=None, T_0=lambda r: 1):

        self.rng = np.random.default_rng()

        self.num_bandits = num_bandits
        # Learner com L maiusculo pois deve ser uma classe
        self.Learner = Learner # keeping reference to the Learner class to create new balls during the run

        self.Learner_kwargs = {**Learner_kwargs, 'num_bandits':num_bandits}

        if context_space == None:
            # First dimension is the depth, second is the size
            # Search space is mapped into these intervals so we can know exactly how to calculate a normalized distance
            context_space = Context_Space(np.linspace(0, 10, 11), np.linspace(0, 110, 111))
            # must contain all possible values 
        self.context_space = context_space

        # Collection of balls in the context space, initialized with one ball that covers the whole context set
        self.A = [Ball(center=np.array([d[(len(d)-1)//2] for d in self.context_space.dimensions]),
                       radius=1.0,
                       Learner=self.Learner,
                       Learner_kwargs=self.Learner_kwargs, 
                       id=1)]
        
        # Set of active balls that are not full
        self.Astar =[self.A[0] ]

        self.T_0 = lambda r: 1000

        self.update_queue = []

        # Store learner status when the update function is called
        self.pull_history = {
            c:[] for c in ['t', 'arm idx', 'reward', 'success', 'update', 'delta error',
                           # Context stuff
                           'gen',
                           'ball_id', 'ball_radius', 'ball_center', 'ball_center_raw',
                           'prg_depth', 'prg_size', 'ball_active', 'distance'] } 
        
    # I'll be updating the learner in this step
    # Aqui que eu tenho que implementar o learner, e o log tem que ser baseado em algo parecido com o listener (salvar s[o o basico de informacao msm])
    def choose_arm(self, **context):
        # TODO: stop doing hardcored access to context information
        x = [context['prg_depth'], context['prg_size']]
        
        # find set of relevants. if not empty, use a random one
        # else, create new ball
        relevant = [B for B in self.Astar if self.context_space.d(x, B.center)<B.radius]
        
        B = None
        if len(relevant)>0:
            B = self.rng.choice(relevant)
        else:
            # the minimum radius value between all balls that contained x_t in the history of the learner
            r = np.min([B.radius for B in self.A if self.context_space.d(x, B.center)<B.radius])
            B = Ball(
                np.array(x), #self.context_space._get_context_set(x), 
                r/2, 
                Learner=self.Learner, 
                Learner_kwargs=self.Learner_kwargs, 
                id=len(self.A)+1)
            
            self.A.append(B)
            self.Astar.append(B)

        self.update_queue.append(B)
        arm = B.choose_arm(**context)
        
        return arm

    def log(self, arm_idx, delta_costs, success, **context):

        assert len(self.update_queue)>0, "trying to update without pulling before"

        self.last_B_ = self.update_queue[0] 
        self.update_queue = self.update_queue[1:]

        # It's not used to make decisions, but we still need to keep track
        reward = self.last_B_.learner._calc_reward(delta_costs)

        # These are global information about arm choices. each individual learner
        # also has it's own (and since it can be inactived, we also have several balls to 
        # gather information).
        self.pull_history['t'].append( len(self.pull_history['t']) )
        self.pull_history['arm idx'].append( arm_idx )
        self.pull_history['reward'].append( reward )
        self.pull_history['success'].append( success )
        self.pull_history['delta error'].append( self.last_B_.learner.weights*delta_costs )

        self.pull_history['ball_id'].append( self.last_B_.id )
        self.pull_history['ball_radius'].append( self.last_B_.radius )
        self.pull_history['ball_active'].append( self.last_B_.active )
        self.pull_history['prg_depth'].append( context['prg_depth'] )
        self.pull_history['prg_size'].append( context['prg_size'] )
        self.pull_history['gen'].append( context['gen'] )
        self.pull_history['ball_center'].append( self.context_space._get_context_set(self.last_B_.center).tolist() )
        self.pull_history['ball_center_raw'].append( self.last_B_.center.tolist() )
        self.pull_history['distance'].append( self.context_space.d([context['prg_depth'], context['prg_size']], self.last_B_.center) )
        
        if self.last_B_.counter == self.T_0(self.last_B_.radius):
            self.pull_history['update'].append( 1 )
        else:
            self.pull_history['update'].append( 0 )

    def update(self, arm_idx, delta_costs, success=1, **context):
        # TODO: remove context from update function, and remove batch updates 

        # Log will check for an empty queue and also update the last_B_ used here
        self.log(arm_idx, delta_costs, success, **context)

        # Update learner
        self.last_B_.update(arm_idx, delta_costs, success, **context)

        # update ball
        self.last_B_.counter = self.last_B_.counter + 1

        # remove ball if necessary
        if self.last_B_.counter == self.T_0(self.last_B_.radius):
            self.last_B_.active = False
            self.Astar.remove(self.last_B_)
            
from .Pareto_UCB1_Learner import Pareto_UCB1_Learner
class C_Pareto_UCB1_Learner(Contextual_Wrapper):
    def __init__(self, num_bandits, **kwargs):
        kwargs['Learner'] = Pareto_UCB1_Learner # Forcing to use D_UCB1
        super().__init__(num_bandits=num_bandits, **kwargs)

from .D_TS_Learner import D_TS_Learner
class C_D_TS_Learner(Contextual_Wrapper):
    def __init__(self, num_bandits, **kwargs):
        kwargs['Learner'] = D_TS_Learner
        super().__init__(num_bandits=num_bandits, **kwargs)

from .D_UCB1_Learner import D_UCB1_Learner
class C_D_UCB1_Learner(Contextual_Wrapper):
    def __init__(self, num_bandits, **kwargs):
        kwargs['Learner'] = D_UCB1_Learner
        super().__init__(num_bandits=num_bandits, **kwargs)

from .UCB1_Learner import UCB1_Learner
class C_UCB1_Learner(Contextual_Wrapper):
    def __init__(self, num_bandits, **kwargs):
        kwargs['Learner'] = UCB1_Learner
        super().__init__(num_bandits=num_bandits, **kwargs)