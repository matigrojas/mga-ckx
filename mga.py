from typing import TypeVar, List
import numpy as np

S = TypeVar('S')
R = TypeVar('R')

class MicroGAFFNN():

    def __init__(self,
                 problem,
                 mutation,
                 crossover,
                 selection,
                 offspring_population_size: int = 6,
                 population_size: int = 5):
        self.problem=problem
        
        self.population_size=population_size
        self.offspring_population_size=offspring_population_size
        
        self.mutation=mutation
        self.crossover=crossover
        self.selection=selection
        
        self.best_solution = None
        self.evaluations = 0
        
        self.start_computing_time = 0
        self.total_computing_time = 0

    def init_progress(self) -> None:
        self.evaluations = self.population_size

        n = self.problem.get_input_size()
        weights_l1 = (n**2)*2+n
        weights_l2 = n*2+1
        biases = n*2+2
        self.crossover_operator.set_network_architecture(weights_l1,weights_l2,biases)

    def selection(self, population: List[S]):
        mating_population = []

        for i in range(self.mating_pool_size):
            solution = self.selection_operator.execute(population)
            mating_population.append(solution)

        return mating_population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        number_of_parents_to_combine = self.crossover_operator.get_number_of_parents()

        #if len(mating_population) % number_of_parents_to_combine != 0:
        #    raise Exception('Wrong number of parents')

        offspring_population = []
        for i in range(0, self.offspring_population_size, 2):
            parents = []
            for j in range(2):
                parents.append(mating_population[i + j])
            if len(parents) < number_of_parents_to_combine:
                parents.append(self.solutions[0])

            offspring = self.crossover_operator.execute(parents)

            for solution in offspring:
                self.mutation_operator.execute(solution)
                offspring_population.append(solution)
                if len(offspring_population) >= self.offspring_population_size:
                    break

        return offspring_population

    def is_nominal_convergence(self, solutions) -> bool:
        acum_mean = 0.0
        for sol in solutions[1:]:
            acum_mean += np.mean(np.abs(np.array(solutions[0].variables) - np.array(sol.variables)))
        return True if acum_mean <= 0.001 else False

    def set_best_solution(self,best_solution):
        self.best_solution = best_solution

    def create_initial_solutions(self) -> List[S]:
        if self.best_solution is not None:
            population = [self.best_solution]
            population.extend([self.population_generator.new(self.problem)
                for _ in range(self.population_size-1)])
        else:
            population = [self.population_generator.new(self.problem)
                for _ in range(self.population_size)]
        return population


    def step(self):
        if not self.termination_criterion.is_met:
            if hasattr(self.mutation_operator, 'set_no_diff_expr'):
                self.mutation_operator.set_no_diff_expr(self.problem.no_diff_expr)

            mating_population = self.selection(self.solutions)
            offspring_population = self.reproduction(mating_population)
            offspring_population = self.evaluate(offspring_population)
            self.evaluations += self.offspring_population_size
            self.update_progress()

            self.solutions = self.replacement(self.solutions, offspring_population)

            if not self.termination_criterion.is_met:
                if self.is_nominal_convergence(self.solutions):
                    self.solutions = self.reset_population(self.solutions)

    def reset_population(self, solutions) -> List[S]:
        solutions = [solutions[0]]
        if not self.termination_criterion.is_met:
            for _ in range(1, self.population_size):
                if not self.termination_criterion.is_met:
                    solutions.append(self.evaluate([self.population_generator.new(self.problem)])[0])
                    self.evaluations += 1
                    self.update_progress()
                else:
                    break
        return solutions