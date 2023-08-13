import os
import random

import numpy as np
import neurolab as nl

from solution import FloatSolution

class ffnnKFold():

    def __init__(self,route_dataset:str = None, dataset_name:str = None, limits: (float,float) = (-1.0,1.0),
                 number_of_folds: int = 10):
        
        self.number_of_folds = number_of_folds
        self.route_dataset = route_dataset
        self.dateset_name = dataset_name
        self.train_samples,self.predicted_samples = self.load_dataset(self.route_dataset,self.dateset_name)

        fl = np.shape(self.train_samples[0])[1]  # First Layer
        sl = (fl * 2) + 1

        size_layers = [fl, sl, 1]

        if size_layers is None:
            size_layers = []
            self.number_of_variables = 0

        datos = [x * y for x, y in zip(size_layers[:-1], size_layers[1:])]  # calculo el total de pesos entre todas las capas
        self.total_weights = np.sum(datos)
        self.total_bias = np.sum(size_layers[1:])
        self.number_of_objectives = 1
        self.number_of_variables = np.sum(size_layers[1:]) + np.sum(datos)#sumo cuantos bias y pesos va a haber, eso da el tamaño de la solución
        self.number_of_constraints = 0

        self.num_layers = len(size_layers)
        self.size_layers = size_layers
        self.net = nl.net.newff([[0,1]]*self.size_layers[0],[self.size_layers[1],1])

        self.obj_directions = [-1,]
        self.obj_labels = ['f(x)']

        self.lower_bound = [limits[0] for _ in range(self.number_of_variables)]
        self.upper_bound = [limits[1] for _ in range(self.number_of_variables)]

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.variables = \
            [random.uniform(self.lower_bound[i] * 1.0, self.upper_bound[i] * 1.0) for i in
             range(self.number_of_variables)]

        return new_solution

    def load_dataset(self,route_dataset, dataset_name):
        train_input = []
        train_output = []
        for i in range(1,self.number_of_folds+1):
            # Make the dataset route
            train_dataset = dataset_name + f"Train_{i}.csv"
            data_train = os.path.join(route_dataset,train_dataset)

            # Read the dataset
            dataset_train = np.loadtxt(open(data_train, "rb"), delimiter=",", skiprows=0)

            train_input.append(dataset_train[:,:-1])
            train_output.append(dataset_train[:,-1])

        return train_input, train_output


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        numInputs = np.shape(self.train_samples[0])[1]  # number of inputs

        # number of hidden neurons
        HiddenNeurons = self.net.layers[0].np['b'][:].shape[0]

        split1 = HiddenNeurons * numInputs
        split2 = split1 + HiddenNeurons
        split3 = split2 + HiddenNeurons

        input_w = np.array(solution.variables[0:split1]).reshape(HiddenNeurons,numInputs)
        layer_w = np.array(solution.variables[split1:split2]).reshape(1,HiddenNeurons)
        input_bias = np.array(solution.variables[split2:split3]).reshape(1,HiddenNeurons)
        bias_output = solution.variables[split3:split3+1]

        self.net.layers[0].np['w'][:] = input_w
        self.net.layers[1].np['w'][:] = layer_w
        self.net.layers[0].np['b'][:] = input_bias
        self.net.layers[1].np['b'][:] = bias_output

        score = 0.0

        for k in range(self.number_of_folds):
            pred = self.net.sim(self.train_samples[k]).reshape(len(self.train_samples[k]))
            score += ((pred - self.predicted_samples[k]) ** 2).mean(axis=None)

        solution.objectives[0] = score/self.number_of_folds

        return solution

    def get_name(self) -> str:
        return 'FFNN KFOLD'

    def get_input_size(self):
        return np.shape(self.train_samples[0])[1]

    def get_dataset_name(self) -> str:
        return self.dateset_name