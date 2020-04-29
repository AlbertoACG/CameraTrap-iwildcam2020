# Author: Fagner Cunha

"""Simple genetic algorithm lib

Based on:
https://github.com/NicolleLouis/geneticAlgorithm/

"""

from __future__ import print_function
from __future__ import division

import random

class PopulationGenerator(object):
    """Abstract class used to provide data to the GA class.
    
    """
    def generateIndividual(self):
        """Creates a individual wich is a random solution for a problem.
        
        """
        raise NotImplementedError
    
    def fitnessIndividual(self, individual):
        """Evaluates how well a individual is in solving the problem.
        
        """
        raise NotImplementedError
    
    def createChild(self, individual1, individual2):
        """Creates a new individual based on two provided ones.
        
        """
        raise NotImplementedError
    
    def mutateIndividual(self, individual):
        """Mutate a solution in order to explore the solution space.
        
        """
        raise NotImplementedError


class GeneticAlgorithmRunner(object):
    """A class to execute a simple genetic algorith.
    
    #Arguments
        populationGenerator: A object of a subclass of PopulationGenerator wich 
            provides all the operations specific of each problem.
        population_size: number of individuals in each generation.
        best_sample: number of top ranked individuals used to generate the next
            generation.
        lucky_few: number of individuals chosen randomly to generate the next
            generation.
        number_of_childs: number os individuals created for each couple of
            individuals selected to generate the next generation.
        number_of_generations: number of iterations executed until evaluates the
            best individuals
        chance_of_mutation: percentage chance of a mutation hapen in a child.
        verbose: whether it show some log output.
         
    """
    
    def __init__(self,
                populationGenerator,
                population_size = 100,
                best_sample = 20,
                lucky_few = 20,
                number_of_childs = 5,
                number_of_generations = 100,
                chance_of_mutation = 5,
                verbose = True):
        
        self.populationGenerator = populationGenerator
        
        if ((best_sample + lucky_few) / 2 * number_of_childs != population_size):
            raise ValueError("Population size is not stable.")
        
        self.population_size = population_size
        self.best_sample = best_sample
        self.lucky_few = lucky_few
        self.number_of_childs = number_of_childs
        self.number_of_generations = number_of_generations
        self.chance_of_mutation = chance_of_mutation
        self.verbose = verbose
        
        self.generations = []
        self.best_individuals = []
        self.current_generation = None
    
    def _generate_first_population(self):
        self.generations = []
        self.best_individuals = []
        population = []
        
        if self.verbose:
            print("Generating first population...")
        
        for i in range(self.population_size):
            population.append(self.populationGenerator.generateIndividual())
        
        self.generations.append(population)
        self.current_generation = 0
        
    def _rank_population(self, population_index):
        population_perf = []
        
        for individual in self.generations[population_index]:
            population_perf += [self.populationGenerator.fitnessIndividual(individual)]
        
        return [x for _,x in sorted(zip(population_perf,list(range(self.population_size))), reverse=False)]
    
    def _select_from_population(self, ranked_population_indexes):
        breeders = []
        
        for i in range(self.best_sample):
            breeders.append(ranked_population_indexes[i])
        
        for i in range(self.lucky_few):
            breeders.append(random.choice(ranked_population_indexes))
        
        random.shuffle(breeders)
        
        return breeders
    
    def _create_children(self, breeders):
        next_population = []
        
        for i in range(int(len(breeders)/2)):
            for j in range(self.number_of_childs):
                next_population.append(self.populationGenerator.createChild(
                    self.generations[self.current_generation][breeders[i]],
                    self.generations[self.current_generation][breeders[len(breeders) -1 -i]]))
        
        return next_population
    
    def _mutate_population(self, population):
        for i in range(self.population_size):
            if random.random() * 100 < self.chance_of_mutation:
                population[i] = self.populationGenerator.mutateIndividual(population[i])
        
        return population
    
    def _next_generation(self):
        ranked_population = self._rank_population(self.current_generation)
        
        #save best individual for the current generation
        self.best_individuals += [self.generations[self.current_generation][ranked_population[0]]]
        
        breeders = self._select_from_population(ranked_population)
        next_population = self._create_children(breeders)
        next_population = self._mutate_population(next_population)
        
        self.generations.append(next_population)
        self.current_generation += 1
        
    def run(self):
        """Runs the genetic algorithm.
        
        #Arguments
            verbose: if the process should show some log during the process.
        
        """

        self._generate_first_population()
        for i in range(self.number_of_generations):
            if self.verbose:
                print("Generation: %s..." % (i), end='\r')
            self._next_generation()

    def get_best_individual(self):
        """Get the individual with best peformance along of all generations
        
        #Returns
            The best individual
        """
        
        best_individual = 0
        min_mse = 100000
        
        for individual in self.best_individuals:
            mse = self.populationGenerator.fitnessIndividual(individual)
            
            if mse < min_mse:
                min_mse = mse
                best_individual = individual
                
                if self.verbose:
                    print("Current minimum MSE: %f" % (min_mse), end='\r')
        
        return best_individual