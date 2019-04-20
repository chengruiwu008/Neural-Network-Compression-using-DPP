from NeuralNet import NeuralNet

net = NeuralNet("../model/simple_dnn_dict.pkl")

# print(net)

neurons_to_del = {
    "dense1": [0,2,4,9,229],
    "dense2": [39,28,91,12]
}

net.delete_neurons(neurons_to_del=neurons_to_del)

print(net)

model = net.to_torch()

print(model.state_dict())
