import random
import copy
import numpy as np
from solution import FloatSolution

"""
.. module:: ckx
   :platform: Unix, Windows
   :synopsis: Module implementing the CKX Operator.

.. moduleauthor:: Matias G. Rojas, Ana Carolina Olivera, Jessica Andrea Carballido, Pablo Javier Vidal
"""

class CrossKnowledgeCrossover():

    def __init__(self, probability: float = 1.0):
        self.weights_l1 = 0
        self.weights_l2 = 0
        self.biases = 0
        self.probability=probability

    def execute(self, parents: [FloatSolution]) -> [FloatSolution]:
        child_1 = copy.deepcopy(parents[0])
        child_2 = copy.deepcopy(parents[1])
        var = (1+random.uniform(-.01,.01))

        if random.random() <= self.probability:
            part_to_cross = random.sample([0,1,2],k=2)
            for part in part_to_cross:
                if part == 0:
                    aux = np.array(child_1.variables[:self.weights_l1]) * var
                    child_1.variables[:self.weights_l1] = np.array(child_2.variables[:self.weights_l1]) * var
                    child_2.variables[:self.weights_l1] = aux
                elif part == 1:
                    aux = np.array(child_1.variables[self.weights_l1:self.weights_l2]) * var
                    child_1.variables[self.weights_l1:self.weights_l2] = np.array(child_2.variables[self.weights_l1:self.weights_l2]) * var
                    child_2.variables[self.weights_l1:self.weights_l2] = aux
                else:
                    aux = np.array(child_1.variables[self.weights_l2:]) * var
                    child_1.variables[self.weights_l2:] = np.array(child_2.variables[self.weights_l2:]) * var
                    child_2.variables[self.weights_l2:] = aux

        return [child_1, child_2]

    def set_network_architecture(self,weights_l1, weights_l2, biases):
        self.weights_l1 = weights_l1
        self.weights_l2 = weights_l1 + weights_l2
        self.biases = weights_l1 + weights_l2 + biases

    def set_current_evaluation(self,evaluation: int):
        self.evaluations = evaluation

    def get_number_of_parents(self) -> int:
        return 2
    def get_number_of_children(self) -> int:
        return 2

    def get_name(self) -> str:
        return 'Crossed Knowledge Crossover (CKX)'
    
