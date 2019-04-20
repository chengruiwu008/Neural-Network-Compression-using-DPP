# in this file, a class will be defined, which can load from a pytorch/tensorflow model/list of matrix
# update/delete certain neurons based on the input
# and export the modified networks back into pytorch or tensorflow
# the data structure to be used will be a ordered dict key will be the name of the layer
# values will be the matrix of the layer
# the connection of the layers will be represented as a simple list

import torch
import collections
import numpy as np
from pydpp.dpp import DPP
import json

from simple_dnn import Net

# model = Net()
# model.load_state_dict(torch.load("./model/simple_dnn_dict.pkl"))
# model.eval()


class Neuron:
    def __init__(self):
        self.layer = None
        self.position = None


class Layer:
    def __init__(self, name, matrix=None, bias=None):
        self.name = name
        self.prev = None
        self.next = None
        self.matrix = matrix  # a numpy array
        self.bias = bias  # a numpy array
        self.input_dim = np.shape(self.matrix)[1]
        self.dim = np.shape(self.matrix)[0]

    def __str__(self):
        res = "input dimension: " + str(self.input_dim) + "\n"
        res += "output dimension: " + str(self.dim) + "\n"
        res += "weights: " + str(self.matrix) + "\n"
        res += "bias: " + str(self.bias)
        return res

    def delete_action(self, list_of_position=None):
        if not list_of_position:
            print("Nothing to delete in layer %s" % self.name)
            return None
        self.matrix = np.delete(self.matrix, list_of_position, axis=0)
        self.bias = np.delete(self.bias, list_of_position)
        self.dim = np.shape(self.matrix)[0]

    def delete_next_action(self, list_of_position=None):
        if not list_of_position:
            print("Nothing to delete in layer %s" % self.name)
            return None
        self.matrix = np.delete(self.matrix, list_of_position, axis=1)


class NeuralNet:

    def __init__(self, model_dir):
        # load the model from a pickled torch NN
        self.model_dir = model_dir
        self.model = Net()
        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()
        weights = self.model.state_dict()
        # print(weights)
        self.layers = collections.OrderedDict()
        last_layer = None

        # initialize the Layers and store in a OrderedDict to make sure we first access the first element
        for key in weights:
            # print(key)
            lay, wb = key.split(".")
            if wb == "weight":
                if last_layer:
                    self.layers[last_layer].next = lay
                layer = Layer(key, weights[key].numpy())
                layer.prev = last_layer
                last_layer = lay
                self.layers[lay] = layer
            elif wb == "bias":
                self.layers[lay].bias = weights[key].numpy()

    def __str__(self):
        res = "NN model:" + "\n"
        for key in self.layers:
            res += key + ":" + str(self.layers[key]) + "\n"
        return res

    def to_torch(self):
        state_dict = collections.OrderedDict()
        size_list = []
        for layer in self.layers:
            state_dict[layer + "." + "weight"] = torch.Tensor(self.layers[layer].matrix)
            state_dict[layer + "." + "bias"] = torch.Tensor(self.layers[layer].bias)
            if not size_list:
                size_list.append(self.layers[layer].input_dim)
            size_list.append(self.layers[layer].dim)
        m = Net(size_list)
        m.load_state_dict(state_dict)
        return m

    def delete_neurons(self, neurons_to_del=None):
        '''
        :param neurons_to_del: a dict of list of neurons to be deleted (outer dict is layer, the inner list is position)
        :return: None
        '''
        if not neurons_to_del:
            print("Nothing to delete!")
            return None
        for layer in neurons_to_del:
            self.layers[layer].delete_action(neurons_to_del[layer])
            if self.layers[layer].next:
                self.layers[self.layers[layer].next].delete_next_action(neurons_to_del[layer])

    def dpp(self, layer, keep):
        '''
        :param layer: the layer to do the dpp
        :param keep: k of k-dpp, number of neurons to keep
        :return: None, just the action of delete neurons
        '''


        pass

