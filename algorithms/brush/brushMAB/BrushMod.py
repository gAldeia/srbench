# hacked brush version that can have a learner
import sys

from brush.estimator import BrushEstimator
import _brush # Brush backend
from sklearn.base import RegressorMixin
import numpy as np
import pandas as pd
from deap import base, creator, tools
from brush.deap_api import DeapIndividual 

from .nsga2_deap import nsga2
from .nsga2island_deap import nsga2island

class BrushEstimatorMod(BrushEstimator): # Modifying brush estimator.
    # Combining crossover and mutation inside a single function, and making it
    # share information with the cpp backend to control and see the mutation results
    def __init__(self, *,
                 Learner=None,
                 mab_batch_size=1,
                 handle_variation_failure=False, 
                 algorithm='nsga2',
                 **kwargs):
        
        super().__init__(**kwargs)

        # All custom stuff must end with an '_' (so get/set params will not try
        # to use them as a parameter for the Brush's C++ backend)

        self.algorithm = algorithm

        self.write_mutation_trace = True

        # Dict used to update brush's C++ backend (only `mutation_options` is
        # being changed, but we need to provide the whole set of parameters)
        self.params_ = self.get_params()

        # mutations optimized by the learner. Learner arms correspond to
        # these mutations in the order they appear here
        self.mutations_ = ['point', 'insert', 'delete', 'subtree', 'toggle_weight_on', 'toggle_weight_off', 'crossover'] # crossover needs to be the last one

        # Crossover will be part of our mutation set (since it never happens together with a mutation).
        # We'll include it in mutations_, and set default weights to be uniformly
        # distributed (1/len(mutations_)) for all mutations
        # self.mutations_.append('crossover')
    
        # We have 6 different mutations, and the learner will learn to choose
        # between these options by maximizing the reward when using each one
        if Learner is not None:
            self.learner_ = Learner(len(self.mutations_))

            if self.learner_.__class__.__name__ != "Listener":
                # Crossover is handled as a mutation and prob will be selected by learner.
                # Listener cannot select prob, so we must leave cx_prob as it is to avoid having no cx
                self.cx_prob = 0.0 
        else:
            self.learner_ = None

        # Whether the learner should update after each mutation, or if it should
        # update only after a certain number of evaluations.
        # Otherwise, it will
        # store all rewards in gen_rewards_ (which is reseted at the beggining
        # of every generation) and do a batch of updates only after finishing
        # mutating the solutions.
        self.mab_batch_size_    = mab_batch_size #self.pop_size
        self.mab_batch_rewards_ = [] #note: not to confuse with batch learning
        self.handle_variation_failure_ = handle_variation_failure

    def _mutate_with_cross(self, ind1, ind2, rnd_flt, gen):
        # Overriding the mutation so it updates our sampling method. Doing the
        # logic on the python-side for now.

        # Creating a wrapper for mutation to be able to control what is happening
        # in the C++ code (this should be prettier in a future implementation)
        
        #if not ind1.fitness.valid:
            # learner will use this information. We need to make sure it was calculated
            #ind1.fitness.values = self.toolbox_.evaluate(ind1)
        
        context = { # In case someone needs to use it
            'gen'       : gen, # So the MABs can know which generation we are. Also useful for logging
            'arms'      : self.mutations_,
            'max_size'  : self.max_size,
            'max_depth' : self.max_depth,
            'prg_size'  : ind1.prg.size(),
            'prg_depth' : ind1.prg.depth()}
        
        # Choosing (in python side) whats going to happen in this mutation call
        attempts = 0
        while attempts < 3:
            attempts = attempts + 1
            crossover = False
            arm_mutation_idx = len(self.mutations_)-1
            if self.learner_ is not None:
                if self.learner_.__class__.__name__ != "Listener": # Listener cannot choose
                    # Information to be used as context
                    arm_mutation_idx = self.learner_.choose_arm(**context)

                    assert 0 <= arm_mutation_idx < len(self.mutations_), \
                        "Learner failed to chose a valid mutation option"
                    
                    if arm_mutation_idx==len(self.mutations_)-1: #going to crossover instead of mutation
                        crossover = True
                    else:
                        # Forcing the c++ backend to select the desired mutation
                        for i, m in enumerate(self.mutations_[:-1]): # Ignoring crossover if its the case
                            self.params_['mutation_options'][m] = 0.0 if i != arm_mutation_idx else 1.0
                else:
                    if rnd_flt() < self.cx_prob:
                        crossover = True
                    else:
                        arm_mutation_idx = -1
            else: # Mutation may happen
                if rnd_flt() < self.cx_prob:
                    crossover = True

            # Aplying it
            if crossover:
                assert arm_mutation_idx==len(self.mutations_)-1, \
                    f"Crossover called with wrong arm number. Expected {len(self.mutations_)-1}, but got {arm_mutation_idx}"

                mutation_idx = arm_mutation_idx
                opt = None
                opt, _ = self._crossover(ind1, ind2)

                # avoid inserting empty solutions
                if opt is not None:
                    offspring = opt
                    offspring.fitness.values = self.toolbox_.evaluate(offspring)

                    if self.learner_ is not None:
                        reward = None

                        # Will raise an error if one of the fitness was not calculated
                        delta_fitness = np.subtract(ind1.fitness.values, offspring.fitness.values)

                        # It will be a vector of variations on the objective functions
                        reward = delta_fitness

                        # if not ignore_this_time:
                        #     self.mab_batch_rewards_.append( (mutation_idx, reward) )

                        self.mab_batch_rewards_.append( (mutation_idx, reward, 1, context) )
                        
                        if len(self.mab_batch_rewards_) >= self.mab_batch_size_:
                            for (mutation_idx, reward, success, context) in self.mab_batch_rewards_:
                                self.learner_.update(mutation_idx, reward, success, **context)
                            self.mab_batch_rewards_ = []
                            
                    return offspring
                else:
                    if self.learner_ is not None:
                        reward = np.array([0 for _ in range(len(ind1.fitness.values))])
                        if self.handle_variation_failure_:
                            self.learner_.log(mutation_idx, reward, 0, **context)
                        else:
                            # Give a null reward to indicate that mutation failed.
                            # Third term of the tuple indicates success=0 for the
                            # update functions (default is 1). We use numeric values to plot
                            # the cummulative sum in base_learner plot functions
                            self.mab_batch_rewards_.append( (mutation_idx, reward, 0, context )) 

                            if len(self.mab_batch_rewards_) >= self.mab_batch_size_:
                                for (mutation_idx, reward, success, context) in self.mab_batch_rewards_:
                                    self.learner_.update(mutation_idx, reward, success, **context)
                                self.mab_batch_rewards_ = []
                        
                    if self.handle_variation_failure_:
                        continue 
                    else:
                        return None
                    
            else: #mutation happened. Just need to see what were going to do with this information
                opt = None
                mutation_idx = -1

                _brush.set_params(self.params_)
                opt = ind1.prg.mutate()

                mutation = _brush.get_params()['mutation_trace']['mutation']
                mutation_idx = self.mutations_.index(mutation)

                # If the learner tried to force a mutation, it should be valid in the trace
                if (self.learner_ is not None) and (self.learner_.__class__.__name__ != "Listener"): 
                    assert mutation_idx == arm_mutation_idx, f"Learner selected mutation {arm_mutation_idx}, but brush used {mutation_idx}"

                # print(opt)
                if opt is not None:
                    offspring = creator.Individual(opt)
                    offspring.fitness.values = self.toolbox_.evaluate(offspring)

                    # print("mutation")
                    # print(ind1.prg.get_model())
                    # print(offspring.prg.get_model())

                    # print("new individual was created successfully")
                    # print(offspring.fitness, offspring.fitness.valid, offspring.prg.get_model())
                    
                    if self.learner_ is not None:
                        # print(offspring.fitness, offspring.fitness.valid, offspring.prg.get_model())
                        
                        reward = None
                    
                        # Will raise an error if one of the fitness was not calculated
                        delta_fitness = np.subtract(ind1.fitness.values, offspring.fitness.values)

                        # It will be a vector of variations on the objective functions
                        reward = delta_fitness

                        # if not ignore_this_time:
                        #     self.mab_batch_rewards_.append( (mutation_idx, reward) )

                        self.mab_batch_rewards_.append( (mutation_idx, reward, 1, context) )
                        if len(self.mab_batch_rewards_) >= self.mab_batch_size_:
                            for (mutation_idx, reward, success, context) in self.mab_batch_rewards_:
                                self.learner_.update(mutation_idx, reward, success, **context)
                            self.mab_batch_rewards_ = []
                            
                    return offspring
                else:
                    if (self.learner_ is not None):
                        reward = np.array([0 for _ in range(len(ind1.fitness.values))])
                        if self.handle_variation_failure_:
                            self.learner_.log(mutation_idx, reward, 0, **context)
                        else:    
                            self.mab_batch_rewards_.append( (mutation_idx, reward, 0, context )) 

                            if len(self.mab_batch_rewards_) >= self.mab_batch_size_:
                                for (mutation_idx, reward, success, context) in self.mab_batch_rewards_:
                                    self.learner_.update(mutation_idx, reward, success, **context)
                                self.mab_batch_rewards_ = []
                    
                    if self.handle_variation_failure_:
                        continue 
                    else:
                        return None
            

    def _setup_toolbox(self, data_train, data_validation):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # creator.create is used to "create new functions", and takes at least
        # 2 arguments: the name of the newly created class and a base class

        # Minimizing/maximizing problem: negative/positive weight, respectively.
        # Our classification is using the error as a metric
        # Comparing fitnesses: https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)

        # create Individual class, inheriting from self.Individual with a fitness attribute
        creator.create("Individual", DeapIndividual, fitness=creator.FitnessMulti)  
        
        toolbox.register("Clone", lambda ind: creator.Individual(ind.prg.copy()))
        
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate_with_cross)

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        #toolbox.register("select", tools.selAutomaticEpsilonLexicase)
        if self.algorithm=="nsga2" or self.algorithm=="nsga2island":
            toolbox.register("select", tools.selTournamentDCD) 
            toolbox.register("survive", tools.selNSGA2)
        elif self.algorithm=="ga" or self.algorithm=="gaisland":
            toolbox.register("select", tools.selTournament, tournsize=3) 
            def offspring(pop, MU): return pop[-MU:] # not perfect in case all variation attempts failed
            toolbox.register("survive", offspring)

        # toolbox.population will return a list of elements by calling toolbox.individual
        toolbox.register("createRandom", self._make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.createRandom)

        toolbox.register("get_objectives", lambda: self.objectives)
        toolbox.register("getBatch", data_train.get_batch)
        toolbox.register("evaluate", self._fitness_function, data=data_train)
        toolbox.register("evaluateValidation", self._fitness_validation, data=data_validation)

        return toolbox
    

    def fit(self, X, y):
        _brush.set_params(self.get_params())
        
        if self.random_state is not None:
            _brush.set_random_state(self.random_state)

        self.feature_names_ = []
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()

        self.data_ = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     validation_size=self.validation_size)

        # set n classes if relevant
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

        # Weight of each objective (+ for maximization, - for minimization)
        obj_weight = {
            "error"      : +1.0 if self.mode=="classification" else -1.0,
            "size"       : -1.0,
            "complexity" : -1.0
        }
        self.weights = [obj_weight[w] for w in self.objectives]

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size)
        self.validation_ = self.data_.get_validation_data()

        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        self.search_space_ = _brush.SearchSpace(self.train_, self.functions_)
        self.toolbox_ = self._setup_toolbox(data_train=self.train_, data_validation=self.validation_)

        if self.algorithm=="nsga2island" or self.algorithm=="gaisland":
            self.archive_, self.logbook_ = nsga2island(
                self.toolbox_, self.max_gen, self.pop_size, self.n_islands,
                self.mig_prob, self.cx_prob, 
                (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)
        elif self.algorithm=="nsga2" or self.algorithm=="ga":
            # nsga2 and ga differ in the toolbox
            self.archive_, self.logbook_ = nsga2(
                self.toolbox_, self.max_gen, self.pop_size, self.cx_prob, 
                (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)
            
        final_ind_idx = 0

        # Each individual is a point in the Multi-Objective space. We multiply
        # the fitness by the weights so greater numbers are always better
        points = np.array([self.toolbox_.evaluateValidation(ind) for ind in self.archive_])
        points = points*np.array(self.weights)

        if self.validation_size==0.0:  # Using the multi-criteria decision making on training data
            # Selecting the best estimator using training data
            # (train data==val data if validation_size is set to 0.0)
            # and multi-criteria decision making

            # Normalizing
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            points = (points - min_vals) / (max_vals - min_vals)
            points = np.nan_to_num(points, nan=0)
            
            # Reference should be best value each obj. can have (after normalization)
            reference = np.array([1, 1])

            # closest to the reference
            final_ind_idx = np.argmin( np.linalg.norm(points - reference, axis=1) )
        else: # Best in obj.1 (loss) in validation data
            final_ind_idx = max(
                range(len(points)),
                key=lambda index: (points[index][0], points[index][1]) )

        self.best_estimator_ = self.archive_[final_ind_idx].prg

        if self.verbosity > 0:
            print(f'best model {self.best_estimator_.get_model()}'+
                  f' with size {self.best_estimator_.size()}, '   +
                  f' depth {self.best_estimator_.depth()}, '      +
                  f' and fitness {self.archive_[0].fitness}'      )

        return self


class BrushRegressorMod(BrushEstimatorMod, RegressorMixin):
    def __init__(self, Learner, **kwargs):
        super().__init__(mode='regressor', Learner=Learner, **kwargs)

    def _error(self, ind, data: _brush.Dataset):
        MSE = np.mean( (data.y-ind.prg.predict(data))**2 )
        if not np.isfinite(MSE): # numeric erros, np.nan, +-np.inf
            MSE = np.inf

        return MSE

    def _fitness_validation(self, ind, data: _brush.Dataset):
        # Fitness without fitting the expression, used with validation data

        ind_objectives = {
            "error"     : self._error(ind, data),
            "size"      : ind.prg.size(),
            "complexity": ind.prg.complexity()
        }
        return [ ind_objectives[obj] for obj in self.objectives ]

    def _fitness_function(self, ind, data: _brush.Dataset):
        ind.prg.fit(data)

        return self._fitness_validation(ind, data)

    def _make_individual(self):
        if self.initialization not in ["uniform", "max_size"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'max_size' or 'uniform'. got {self.initialization}")
        
        # No arguments (or zero): brush will use PARAMS passed in set_params.
        # max_size is sampled between 1 and params['max_size'] if zero is provided
        return creator.Individual(
            self.search_space_.make_regressor(
                self.max_depth, (0 if self.initialization=='uniform' else self.max_size))
        )