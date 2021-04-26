# None of this has been tested.

import numpy as np


def initialise_parameters(layer_dimensions, seed=42):
    np.random.seed(seed)
    layers = len(layer_dimensions)
    parameters = {}
    for layer in range(1, layers):
        weights = np.random.randn(layer_dimensions[layer],
                                  layer_dimensions[layer - 1]) * 0.01
        bias = np.zeros((layer_dimensions[layer], 1))
        parameters["L" + str(layer) + "_weights"] = weights
        parameters["L" + str(layer) + "_bias"] = bias
    return parameters


def linear_prediction(weights, input_activation, bias):
    return np.dot(weights.T, input_activation) + bias


def compute_activation(z, activation_function):
    if activation_function == 'relu':
        return np.fmax(0, z)
    elif activation_function == 'sigmoid':
        return 1 / (1 + np.exp(-z))


def compute_cost(predictions, actual_values):
    m = actual_values.shape[1]
    cost = -1/m * np.sum(np.log(predictions)
                         + (1 - actual_values) * np.log(1 - predictions))
    return cost


def compute_linear_gradients(d_loss_wrt_z, weights, input_activation):
    m = d_loss_wrt_z.shape[1]
    d_loss_wrt_weights = 1/m * np.dot(input_activation, d_loss_wrt_z.T)
    d_loss_wrt_bias = 1/m * np.sum(d_loss_wrt_z, axis=1, keepdims=True)
    d_loss_wrt_input_activation = np.dot(weights, d_loss_wrt_z)
    return d_loss_wrt_weights, d_loss_wrt_bias, d_loss_wrt_input_activation


def compute_activation_gradient(output_activation,
                                d_loss_wrt_output_activation,
                                activation_function):
    if activation_function == 'relu':
        pass
    elif activation_function == 'sigmoid':
        d_output_activation_wrt_z = (output_activation
                                     * (1 - output_activation))

    d_loss_wrt_z = d_output_activation_wrt_z * d_loss_wrt_output_activation
    return d_loss_wrt_z


if __name__ == '__main__':
    pass
