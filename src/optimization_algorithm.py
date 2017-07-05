import numpy as np
from node import ParameterNode

class OptimizationAlgorithm:
    def __init__(self, parameter_nodes, learning_rate):
        self.parameter_nodes = parameter_nodes

        # Parameters
        self.learning_rate = learning_rate

    def optimize(self, batch_size=1):
        for i, node in enumerate(self.parameter_nodes):
            direction = self.compute_direction(i, node.acc_dJdw / batch_size)
            node.w -= self.learning_rate * direction
        self.reset_accumulators()

    def compute_direction(self, i, grad):
        raise NotImplementedError()

    def reset_accumulators(self):
        for node in self.parameter_nodes:
            node.reset_accumulator()

class GradientDescent(OptimizationAlgorithm):
    def __init__(self, parameter_nodes, learning_rate):
        OptimizationAlgorithm.__init__(self, parameter_nodes, learning_rate)

    def compute_direction(self, i, grad):
        return grad