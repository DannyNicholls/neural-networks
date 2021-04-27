# None of this has been tested.

import numpy as np


def initialise_model_parameters(layer_dimensions, seed=42):
    np.random.seed(seed)
    number_of_layers = len(layer_dimensions)
    model_parameters = {}
    for layer in range(1, number_of_layers):
        number_of_inputs = layer_dimensions[layer - 1]
        number_of_nodes = layer_dimensions[layer]
        W = np.random.randn(number_of_inputs, number_of_nodes) * 0.01
        b = np.zeros((number_of_nodes, 1))
        model_parameters["W" + str(layer)] = W
        model_parameters["b" + str(layer)] = b
    return model_parameters


def linear_prediction(W, input_activation, b):
    Z = np.dot(W.T, input_activation) + b
    return Z


def compute_activation(Z, activation_function):
    if activation_function == 'relu':
        A = np.fmax(0, Z)
    elif activation_function == 'sigmoid':
        A = 1 / (1 + np.exp(-Z))
    return A


def forward(X, model_parameters):
    number_of_layers = (len(model_parameters) + 2) // 2
    activations = {'A0': X}

    for layer in range(1, number_of_layers):
        W = model_parameters['W' + str(layer)]
        Aprev = activations["A" + str(layer - 1)]
        b = model_parameters['b' + str(layer)]
        Z = linear_prediction(W, Aprev, b)
        activation_function = ('sigmoid' if layer == number_of_layers - 1
                               else 'relu')
        A = compute_activation(Z, activation_function)
        activations["A" + str(layer)] = A

    return activations


def compute_cost(Yhat, Y):
    m = Y.shape[1]
    cost = -1/m * np.sum(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
    dYhat = - (np.divide(Y, Yhat) - np.divide(1 - Y, 1 - Yhat))
    return cost, dYhat


def compute_activation_gradient(A, dA, activation_function):
    dJdA = dA
    if activation_function == 'relu':
        dAdZ = 1 if A > 0 else 0
    elif activation_function == 'sigmoid':
        dAdZ = (A * (1 - A))
    dZ = dAdZ * dJdA
    return dZ


def compute_linear_gradients(dZ, W, Aprev):
    m = dZ.shape[1]
    dW = 1/m * np.dot(Aprev, dZ.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dAprev = np.dot(W, dZ)
    return dW, db, dAprev


def backward(dYhat, model_parameters, activations):
    number_of_layers = len(activations) // 2
    gradients = {'dA' + str(number_of_layers - 1): dYhat}

    for layer in reversed(range(1, number_of_layers)):
        W = model_parameters['W' + str(layer)]
        A = activations['A' + str(layer)]
        activation_function = 'sigmoid' if layer == number_of_layers else 'relu'
        dA = gradients['dA' + str(layer)]
        dZ = compute_activation_gradient(A, dA, activation_function)
        Aprev = activations['A' + str(layer - 1)]
        dW, db, dAprev = compute_linear_gradients(dZ, W, Aprev)

        gradients['dW' + str(layer)] = dW
        gradients['db' + str(layer)] = db
        gradients['dA' + str(layer - 1)] = dAprev

    return gradients


def update_model_parameters(model_parameters, gradients, learning_rate):
    number_of_layers = (len(model_parameters) + 2) // 2

    for layer in range(1, number_of_layers):
        model_parameters['W' + str(layer)] -= (learning_rate
                                               * gradients['W' + str(layer)])
        model_parameters['b' + str(layer)] -= (learning_rate
                                               * gradients['b' + str(layer)])

    return model_parameters


def train(X_train, Y_train, layer_dimensions, number_of_iterations):
    number_of_layers = len(layer_dimensions)
    model_parameters = initialise_model_parameters(layer_dimensions)

    for _ in range(number_of_iterations):
        activations = forward(X_train, model_parameters)
        cost, dYhat = compute_cost(activations['A' + str(number_of_layers - 1)],
                                   Y_train)
        gradients = backward(dYhat, model_parameters, activations)
        model_parameters = update_model_parameters(model_parameters,
                                                   gradients,
                                                   0.1)

    return model_parameters


if __name__ == '__main__':
    predictions = [1 if a > 0.5 else 0 for a in A]