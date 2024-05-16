# hacked brush version that can have a learner
import numpy as np
from deap import tools
from deap.benchmarks.tools import hypervolume
import functools


def nsga2(toolbox, NGEN, MU, CXPB, use_batch, verbosity, rnd_flt):
    # NGEN = 250
    # MU   = 100
    # CXPB = 0.9
    # rnd_flt: random number generator to sample crossover prob

    def calculate_statistics(ind):
        on_train = ind.fitness.values
        on_val   = toolbox.evaluateValidation(ind)

        return (*on_train, *on_val) 

    stats = tools.Statistics(calculate_statistics)

    stats.register("avg", np.nanmean, axis=0)
    stats.register("med", np.nanmedian, axis=0)
    stats.register("std", np.nanstd, axis=0)
    stats.register("min", np.nanmin, axis=0)
    stats.register("max", np.nanmax, axis=0)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'evals', 'best_size'] + \
                     [f"{stat} {partition} O{objective}"
                         for stat in ['avg', 'med', 'std', 'min', 'max']
                         for partition in ['train', 'val']
                         for objective in toolbox.get_objectives()]

    pop = toolbox.population(n=MU)

    fitnesses = toolbox.map(functools.partial(toolbox.evaluate), pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.survive(pop, len(pop))

    record = stats.compile(pop)
    
    # Finding the size (obj2) of the individual with best error (obj1)
    best_size = max( range(len(pop)),
        key=lambda index: ( pop[index].fitness.values[0]*pop[index].fitness.weights[0],
                            pop[index].fitness.values[1]*pop[index].fitness.weights[1]) )
    
    logbook.record(gen=0, evals=len(pop), 
                   best_size=pop[best_size].fitness.values[1], **record)

    if verbosity > 0: 
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, NGEN+1):
        batch = toolbox.getBatch()
        if (use_batch):
            fitnesses = toolbox.map(functools.partial(toolbox.evaluateValidation, data=batch), pop)
        
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

        # Vary the population
        # offspring = tools.selTournamentDCD(pop, len(pop))
        parents = toolbox.select(pop, len(pop))
        # offspring = [toolbox.clone(ind) for ind in offspring]
        offspring = []
        for ind1, ind2 in zip(parents[::2], parents[1::2]):
             # Crossover is actually a mutation, so it may happen inside mutation,
             # and thats why i need two individuals for this (will be captured by the learners)
            off1 = toolbox.mutate(ind1, ind2, rnd_flt, gen)
            off2 = toolbox.mutate(ind2, ind1, rnd_flt, gen) # This custom mutation will always calculate the fitness
            
            if off1 is not None: # Expressions are fitted inside our custom mutation
                if use_batch: # Adjust fitness to the same data as parents
                    off1.fitness.values = toolbox.evaluateValidation(off1, data=batch)
                offspring.extend([off1])

            if off2 is not None:
                if use_batch:
                    off2.fitness.values = toolbox.evaluateValidation(off2, data=batch)
                offspring.extend([off2])

        # line below is already executed inside mutation in brushMod
        # fitnesses = toolbox.map(functools.partial(toolbox.evaluate), offspring)

        # Select the next generation population
        pop = toolbox.survive(pop + offspring, MU)
        
        record = stats.compile(pop)
        
        best_size = max( range(len(pop)),
            key=lambda index: ( pop[index].fitness.values[0]*pop[index].fitness.weights[0],
                                pop[index].fitness.values[1]*pop[index].fitness.weights[1]) )
        
        logbook.record(gen=gen, evals=len(offspring)+(len(pop) if use_batch else 0),
                       best_size=pop[best_size].fitness.values[1], **record)

        if verbosity > 0: 
            print(logbook.stream)

    if verbosity > 0: 
        print("Final population hypervolume is %f" % hypervolume(pop, [1000.0, 50.0]))

    archive = tools.ParetoFront() 
    archive.update(pop)

    return archive, logbook