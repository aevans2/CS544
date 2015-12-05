#start_imports
from random import Random
from time import time
from math import sin
from math import sqrt
from inspyred import ec
from inspyred.ec import terminators
#end_imports


#generator
def generate_schwefel(random, args):
    size = args.get('num_inputs', 10)
    return [random.uniform(-500, 500) for i in range(size)]

#evaluator
def evaluate_schwefel(candidates, args):
    fitness = []
    n = 2
    for cs in candidates:
        fit = 418.9829 * n - sum([(-x * sin(sqrt(abs(x)))) for x in cs])
        fitness.append(fit)
    return fitness

#start_main
rand = Random()
rand.seed(int(time()))
es = ec.ES(rand)
es.terminator = terminators.evaluation_termination
final_pop = es.evolve(generator=generate_schwefel,
                      evaluator=evaluate_schwefel,
                      pop_size=100,
                      maximize=False,
                      bounder=ec.Bounder(-500, 500),
                      max_evaluations=20000,
                      mutation_rate=0.25,
                      num_inputs=2,
                      )
# Sort and print the best individual, who will be at index 0.
final_pop.sort(reverse=True)
print ("Minimum value for f(x): ")
print(final_pop[0].candidate[0:2])
#end_main
