"""
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

The matrix dimensions are aligned such that rows represent nodes and columns
represent examples.  One exception being the weight matrices, for which the
rows represent nodes in layer l - 1 and the columns represent nodes in layer l.

Have a great day!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def initialise_model_parameters(layer_dimensions, seed=42):
    """Return a dictionary of random weight values.  For each layer, l,
    greater than 1, the returned dictionary includes a key, Wl, to a numpy
    array storing weights for that layer, and a key, bl, to a numpy array
    storing a bias vector for that layer."""
    np.random.seed(seed)
    number_of_layers = len(layer_dimensions)
    model_parameters = {}
    for layer in range(1, number_of_layers):
        number_of_inputs = layer_dimensions[layer - 1]
        number_of_nodes = layer_dimensions[layer]

        W = np.random.randn(number_of_inputs, number_of_nodes)
        b = np.random.randn(number_of_nodes, 1)

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
    """Return a dictionary of activation values.  For each layer, l, greater
    than 1, the dictionary inlcudes a key, Al, to numpy array storing the
    activations for that layer."""
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
    """Return cross-entropy cost."""
    m = Y.shape[1]

    # Add an offset to ensure that we do not calculate the log of zero.
    offset = 1e-3
    cost = (-1/m
            * np.sum(np.multiply(Y, np.log(Yhat + offset))
                     + np.multiply((1 - Y), np.log(1 - Yhat + offset))))

    return cost


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


def backward(Y, activations, model_parameters, activation_functions):
    """Return a dictionary of gradients.  For each layer, l, greater than 1,
    the dictionary inlcudes key, dWl, db, dA, to numpy arrays storing the
    gradients for that layer."""
    number_of_layers = len(activations)

    # Calculate the derivative of the cost function with respect to the output
    # layer activation.
    Yhat = activations['A' + str(number_of_layers - 1)]
    offset = 1e-3
    activation_function = activation_functions[number_of_layers - 1]
    if activation_function == 'sigmoid':
        dYhat = (np.divide(1 - Y, 1 - Yhat + offset)
                 - np.divide(Y, Yhat + offset))
    else:
        pass # need to add softmax.

    gradients = {'dA' + str(number_of_layers - 1): dYhat}

    # Propagate the derivative of the cost function with respect to the
    # output layer activation back through the network.
    for layer in reversed(range(1, number_of_layers)):

        # Obtain the model parameters, activations, and gradients required for
        # backward propagation.
        W = model_parameters['W' + str(layer)]
        activation_function = activation_functions[layer]
        A = activations['A' + str(layer)]
        Aprev = activations['A' + str(layer - 1)]
        dA = gradients['dA' + str(layer)]

        # Compute the gradients.
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

    # Some simple error checking to be expanded upon at a later date.  Could
    # automatically arrange data, for example.
    assert len(layer_dimensions) > 1
    assert len(layer_dimensions) == len(activation_functions)
    assert len(Y_train) == layer_dimensions[-1]
    assert len(X_train) == layer_dimensions[0]

    number_of_layers = len(layer_dimensions)
    model_parameters = initialise_model_parameters(layer_dimensions)

    # Forward and backward propagation.
    costs = [[], []]
    for iteration in range(1, number_of_iterations + 1):

        # Forward.
        activations = forward(X_train,
                              model_parameters, activation_functions)
        Yhat = activations['A' + str(number_of_layers - 1)]
        cost = compute_cost(Yhat, Y_train)

        # Backward.
        gradients = backward(Y_train, activations,
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
            if iteration == number_of_iterations:
                fig, ax = plt.subplots()
                ax.plot(costs[0], costs[1], c='hotpink')
                ax.set_title('Hot Pink Loss Curve!')
                ax.set_ylabel('Cost')
                ax.set_xlabel('Iteration')
                fig.show()

    return model_parameters


def test(X_test, Y_test, model_parameters, activation_functions):
    output_layer_number = (len(model_parameters) + 2) // 2 - 1
    activations = forward(X_test, model_parameters, activation_functions)
    Yhat = activations['A' + str(output_layer_number)]
    cost = compute_cost(Yhat, Y_test)
    return Yhat, cost


def or_gate():
    """
    Train a neural network to predict the output of an OR gate.  This is
    useful for testing code for a single classification.
    """
    X = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 1]])
    Y = np.array([[0, 1, 1, 1]])    # X[0] bitwise-OR X[1]

    # Train a model.
    layer_dimensions = [2, 2, 1]
    activation_functions = [None, 'sigmoid', 'sigmoid']
    model_parameters = train(X, Y,
                             layer_dimensions, activation_functions,
                             number_of_iterations=5000, learning_rate=.05,
                             print_cost=True, print_rate=500)

    # Test the model.
    Yhat, _ = test(X, Y, model_parameters, activation_functions)
    predictions = Yhat > 0.5

    # Print the predicted values.
    for i in range(4):
        print(X[1][i], 'or', X[0][i], '=', Y[0][i],
              'for which the model predicts', predictions[0][i].astype(int))


def or_and_xor_gates():
    """
    Train a neural network to predict the output of OR, AND, and XOR gates.
    This is useful for testing code for multi-class classification.
    """
    X = np.array([[0, 1, 0, 1],
                  [0, 0, 1, 1]])
    Y = np.array([[0, 1, 1, 1],     # X[0] bitwise-OR X[1]
                  [0, 0, 0, 1],     # X[0] bitwise-AND X[1]
                  [0, 1, 1, 0]])    # X[0] bitwise-XOR X[1]

    # Train a model.
    layer_dimensions = [2, 4, 3]
    activation_functions = [None, 'relu', 'sigmoid']
    model_parameters = train(X, Y,
                             layer_dimensions, activation_functions,
                             number_of_iterations=100,
                             learning_rate=.005,
                             print_cost=True, print_rate=100)

    # Test the model.
    Yhat, _ = test(X, Y, model_parameters, activation_functions)
    predictions = Yhat > 0.5

    # Print the predicted values.
    print()
    for i in range(4):
        print(X[1][i], 'OR', X[0][i], '==', Y[0][i],
              'for which the model predicts', predictions[0][i].astype(int))
    print()
    for i in range(4):
        print(X[1][i], 'AND', X[0][i], '==', Y[1][i],
              'for which the model predicts', predictions[1][i].astype(int))
    print()
    for i in range(4):
        print(X[1][i], 'XOR', X[0][i], '==', Y[1][i],
              'for which the model predicts', predictions[2][i].astype(int))


def digits():

    # MNIST database.
    file_path_to_csv = '/Users/Danny/Documents/Python/Digits/train.csv'
    data = pd.read_csv(file_path_to_csv)
    data = data.sample(frac=1)

    # Store each training example input in a separate column.
    X_train = data.iloc[0:400, 1:].T
    number_of_features = len(X_train)
    m = len(X_train.iloc[0])

    # In the data I am using, zeros are encoded as 10.
    data['y'].replace(10, 0, inplace=True)

    # One-hot encode the training example y values.
    Y_train = np.array([[0] * 10 for _ in range(m)]).T
    for i in range(m):
        Y_train[data.iloc[i, 0]][i] = 1

    # Plot some examples from the training set.
    plt.figure()
    for i in range(1, 10):
        example_X = np.array(X_train.iloc[:, i])
        example_y = data.iloc[i, 0]
        plt.subplot(3, 3, i)
        plt.imshow(example_X.reshape(20, 20).T)
        plt.title(data.iloc[i, 0])
        plt.yticks([])
        plt.xticks([])
        plt.xlabel(''.join([str(num) for num in Y_train[:, i]]))
    plt.suptitle('Some examples from the training set', fontsize=16)
    plt.subplots_adjust(hspace=0.7)

    # Train a model.
    layer_dimensions = [number_of_features, 25, 10]
    activation_functions = [None, 'relu', 'sigmoid']
    np.seterr(all='raise')
    model_parameters = train(X_train, Y_train,
                             layer_dimensions, activation_functions,
                             number_of_iterations=100, learning_rate=0.1,
                             print_cost=True, print_rate=1)

    # Test the model.
    X_test = data.iloc[1:4000, 1:].T
    Y_test = np.array([data.iloc[1:4000, 0]])
    Y_test -= 1

    output_activations, cost = test(X_test, Y_test,
                                    model_parameters, activation_functions)

    predictions = np.argmax(output_activations, axis=0)

    correct = np.sum(predictions == Y_test)
    incorrect = np.sum(predictions != Y_test)
    percent_correct = correct / (correct + incorrect) * 100


if __name__ == '__main__':
    or_and_xor_gates()
