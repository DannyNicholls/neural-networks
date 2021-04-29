'''
I have used the following abbreviations which I hope makes the code easier
to read:

    W      Weights
    b      Bias values
    Z      Linear activations
    A      Non-linear activations
    Aprev  Non-linear activations of the previous layer
    Yhat   Output layer activations
    dW     Differential of the cost function w.r.t. the weights
    db     Differential of the cost function w.r.t. the bias values
    dA     Differential of the cost function w.r.t. the non-linear activations
    dYhat  Differential of the cost function w.r.t. the output layer activation
    dAdZ   Differential of the non-linear activations w.r.t the linear
           activations

Have a great day!
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def forward(X, model_parameters, activation_functions):
    number_of_layers = (len(model_parameters) + 2) // 2
    activations = {'A0': X}

    for layer in range(1, number_of_layers):

        # Obtain parameters required for forward propagation.
        W = model_parameters['W' + str(layer)]
        b = model_parameters['b' + str(layer)]
        Aprev = activations["A" + str(layer - 1)]
        activation_function = activation_functions[layer]

        # Compute activations.
        Z = linear_prediction(W, Aprev, b)
        A = compute_activation(Z, activation_function)

        activations["A" + str(layer)] = A

    return activations


def compute_cost(Yhat, Y):
    m = Y.shape[1]
    zero_offset = 1e-3
    cost = -1/m * np.sum(Y * np.log(Yhat + zero_offset)
                         + (1 - Y) * np.log(1 - Yhat + zero_offset))

    # The following assumes that the output layer is sigmoid.  Need to
    # parameterize this.
    dYhat = - (np.divide(Y, Yhat + zero_offset)
               - np.divide(1 - Y, 1 - Yhat + zero_offset))

    return cost, dYhat


def compute_activation_gradient(A, dA, activation_function):
    dJdA = dA
    if activation_function == 'relu':
        dAdZ = A > 0
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


def backward(dYhat, activations, model_parameters, activation_functions):
    number_of_layers = len(activations)
    gradients = {'dA' + str(number_of_layers - 1): dYhat}
    for layer in reversed(range(1, number_of_layers)):

        # Obtain model parameters, activations, and gradients for backward
        # propagation.
        W = model_parameters['W' + str(layer)]
        activation_function = activation_functions[layer]
        A = activations['A' + str(layer)]
        Aprev = activations['A' + str(layer - 1)]
        dA = gradients['dA' + str(layer)]

        # Compute gradients.
        dZ = compute_activation_gradient(A, dA, activation_function)
        dW, db, dAprev = compute_linear_gradients(dZ, W, Aprev)

        gradients['dW' + str(layer)] = dW
        gradients['db' + str(layer)] = db
        gradients['dA' + str(layer - 1)] = dAprev

    return gradients


def update_model_parameters(model_parameters, gradients, learning_rate):
    number_of_layers = (len(model_parameters) + 2) // 2
    for layer in range(1, number_of_layers):
        model_parameters['W' + str(layer)] -= (learning_rate
                                               * gradients['dW' + str(layer)])
        model_parameters['b' + str(layer)] -= (learning_rate
                                               * gradients['db' + str(layer)])
    return model_parameters


def train(X_train, Y_train,
          layer_dimensions, activation_functions,
          number_of_iterations, learning_rate,
          print_cost=False, print_rate=250):

    number_of_layers = len(layer_dimensions)
    model_parameters = initialise_model_parameters(layer_dimensions)

    # Forward and backward propagation.
    costs = [[], []]
    for iteration in range(1, number_of_iterations + 1):
        activations = forward(X_train, model_parameters, activation_functions)
        output_activations = activations['A' + str(number_of_layers - 1)]
        cost, dYhat = compute_cost(output_activations, Y_train)
        gradients = backward(dYhat, activations,
                             model_parameters, activation_functions)
        model_parameters = update_model_parameters(model_parameters,
                                                   gradients,
                                                   learning_rate)

        # Printing and plotting cost.
        if print_cost:
            costs[0].append(iteration)
            costs[1].append(cost)
            if iteration % print_rate == 0:
                print('Cost at iteration {}: {}'.format(iteration, cost))
            if iteration == number_of_iterations - 1:
                fig, ax = plt.subplots()
                ax.plot(costs[0], costs[1], c='hotpink')
                ax.set_title('Model Training')
                ax.set_ylabel('Cost')
                ax.set_xlabel('Iteration')
                fig.show()

    return model_parameters


def test(X_test, Y_test, model_parameters, activation_functions):
    output_layer_number = (len(model_parameters) + 2) // 2 - 1
    activations = forward(X_test, model_parameters, activation_functions)
    output_activations = activations['A' + str(output_layer_number)].T
    cost, _ = compute_cost(output_activations, Y_test)
    return output_activations, cost


def or_gate():
    # Y is the bitwise-or of X[0] and X[1].
    X = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
    Y = np.array([[0, 1, 1, 1]])

    # Train a model.
    layer_dimensions = [2, 2, 1]
    activation_functions = [None, 'relu', 'sigmoid']
    model_parameters = train(X, Y,
                             layer_dimensions, activation_functions,
                             number_of_iterations=25000, learning_rate=.05,
                             print_cost=True, print_rate=500)

    # Test the model.
    output_activations, cost = test(X, Y,
                                    model_parameters, activation_functions)

    # Print the predicted values.
    print('\nFour cases:')
    for i in range(4):
        print(X[1][i], 'or', X[0][i], '=', Y[0][i],
              'for which the model predicts', *output_activations[i])


def digits():

    # MNIST database.
    file_path_to_train_csv = '/Users/Danny/Documents/Python/Digits/train.csv'
    file_path_to_test_csv = '/Users/Danny/Documents/Python/Digits/test.csv'
    training_data = pd.read_csv(file_path_to_train_csv)
    test_data = pd.read_csv(file_path_to_test_csv)

    # Store each training example input in a separate column.
    X_train = training_data.iloc[1:, 1:].T
    number_of_features = len(X_train)
    m = len(X_train.iloc[0])

    # One-hot encode the training example y values and store each one-hot
    # encoded value in a separate column.
    Y_train = np.array([[0] * 10 for _ in range(m)]).T
    for i in range(m):
        Y_train[training_data.iloc[i, 0]][i] = 1

    # Train a model.
    layer_dimensions = [number_of_features]
    activation_functions = [None]
    model_parameters = train(X_train, Y_train,
                             layer_dimensions, activation_functions,
                             number_of_iterations=10000, learning_rate=0.1,
                             print_cost=True, print_rate=50)

    # Test the model.
    X_test = test_data.iloc[1:, :].T
    Y_test = np.array(test_data.iloc[1:, 0].values.reshape(1, 27999))
    output_activations, cost = test(X_test, Y_test, model_parameters)
    predictions = np.argmax(output_activations, axis=0)


if __name__ == '__main__':
    or_gate()
