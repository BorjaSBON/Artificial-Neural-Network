# Libraries

import pandas as pd
import numpy as np
import random

# Auxiliar Functions

## Generate Weigths
def generateWeights(data, neurons):
    weigths = pd.DataFrame([])

    for layer in range(len(neurons)):
        for neuron in range(neurons[layer]):
            weigth_neuron = []
            
            # The number of weights for echa neuron in the layer 0 is the number of inputs
            # In the other layers, the weights for each neuron are the number of neurons of the previous layer
            if layer == 0:
                n_weights = len(data.columns) + 1
            else:
                n_weights = neurons[layer - 1] + 1
            
            # Generate the weights for the neuron
            weigth_neuron += [layer+1, neuron+1]
            for i in range(n_weights):
                weigth_neuron += [random.uniform(-1,1)]
            
            # Add the weights of the neuron to the final DataFrame
            weigths = pd.concat([weigths, pd.DataFrame(weigth_neuron)], axis=1)
    weights = weigths.transpose()

    # Names of the columns of the DataFrame
    columns = ['Layer','Neuron']
    for i in range(weights.shape[1] - 2):
        columns += ['w' + str(i)]

    # Configure the DataFrame
    weights.columns = columns
    weights.reset_index(drop=True, inplace=True)
    weights[['Layer','Neuron']] = weights[['Layer','Neuron']].astype('int64')

    return weights

## Types of Functions
def sigmoid(value):
    result = 1 / (1 + np.exp(-1 * value))
    return result

def sigmoid_derivate(value):
    result = sigmoid(value) * (1 - sigmoid(value))
    return result

def hyperbolic(value):
    result = (1 - np.exp(-2 * value)) / (1 + np.exp(-2 * value))
    return result

def hyperbolic_derivate(value):
    result = 1 - np.power(hyperbolic(value), 2)
    return result

## Propagate Inputs
def propagateInputs(pattern, neurons, weigths, function):
    outs = pd.DataFrame([], columns=['Layer','Neuron','Value'])
    
    for layer in range(len(neurons)):
        for neuron in range(neurons[layer]):
            weights_neuron = weigths[(weigths['Layer'] == layer + 1)]
            weights_neuron = weights_neuron[(weights_neuron['Neuron'] == neuron + 1)]
            weights_neuron.dropna(axis=1, inplace=True)

            # The weight of the bias (w0)
            net = weights_neuron.iloc[0,2]

            n_weights = weights_neuron.shape[1] - 3
            for i in range(n_weights):
                # If we are int the layer 0, we take in consideration the inputs values
                # If not, we take the previous outputs
                if (layer == 0):
                    net += weights_neuron.iloc[0,2+i+1] * pattern[i]
                else:
                    net += weights_neuron.iloc[0,2+i+1] * outs[outs['Layer'] == layer].iloc[i,2]
            
            # Calculate the outputs depending on the function we use
            if function == 'sigmoid':
                out = pd.DataFrame([[layer + 1, neuron + 1, sigmoid(net)]], columns=['Layer','Neuron','Value'])
            elif function == 'hyperbolic':
                out = pd.DataFrame([[layer + 1, neuron + 1, hyperbolic(net)]], columns=['Layer','Neuron','Value'])
            
            outs = pd.concat([outs, out], axis=0)

    return outs

## Backpropagate Error
def backpropagateError(neurons, desire, outs, weights, function):
    deltas = pd.DataFrame([], columns=['Layer','Neuron','Delta'])

    # Calculate the deltas of the outputs neurons
    for outs_neurons in range(neurons[-1]):
        out = outs[outs['Layer'] == outs['Layer'].max()]
        out = out[out['Neuron'] == outs_neurons+1].iloc[0,2]

        if function == 'sigmoid':
            delta = -1 * (desire[outs_neurons] / out) * out * (1 - out)
        elif function == 'hyperbolic':
            delta = -1 * (desire[outs_neurons] / out) * (1 - np.power(out, 2))

        # Add the deltas of the outputs neurons to the final DataFrame
        delta = pd.DataFrame([outs.iloc[-1,0], outs_neurons+1, delta]).transpose()
        delta.columns=['Layer','Neuron','Delta']
        deltas = pd.concat([deltas, delta])

    # Backpropagate to calculate the deltas of the hidden layers
    for layer in range(len(neurons) - 1):
        layer_i = outs['Layer'].max() - layer - 1

        weight = weights[weights['Layer'] == layer_i + 1]
        delta_post = deltas[deltas['Layer'] == layer_i + 1]
        out_post = outs[outs['Layer'] == layer_i]# + 1]
        
        for neuron in range(neurons[layer_i-1]):
            delta = 0
            for i in range(neurons[layer_i]):
                delta += delta_post.iloc[i,2] * weight.iloc[i,3+neuron]
            
            if function == 'sigmoid':
                delta *= out_post.iloc[neuron,2] * (1 - out_post.iloc[neuron,2])
            elif function == 'hyperbolic':
                delta *= (1 - np.power(out_post.iloc[neuron,2], 2))

            # Add the deltas of the neurons to the final DataFrame
            delta = pd.DataFrame([layer_i, neuron+1, delta]).transpose()
            delta.columns=['Layer','Neuron','Delta']
            deltas = pd.concat([deltas, delta])
    
    return deltas

## Accumulate Change
def accumulateChange(pattern, neurons, outs, deltas):
    gradients = pd.DataFrame([], columns=['Layer','Neuron', 'Weight', 'Gradient'])

    for layer in range(len(neurons)):
        for neuron in range(neurons[layer]):
            deltas_neuron = deltas[deltas['Layer'] == layer + 1]
            deltas_neuron = deltas_neuron[deltas_neuron['Neuron'] == neuron + 1]
            
            # Check if we are in the first layer
            if layer == 0:
                outs_neuron = pd.Series(pattern)
                weights_number = len(outs_neuron) + 1
            else:
                weights_number = neurons[layer-1] + 1
                outs_neuron = outs[outs['Layer'] == layer]
            
            # Calculate the gradients
            gradient = 0
            for i in range(weights_number):
                if layer == 0:
                    if i == 0:
                        gradient = deltas_neuron.iloc[0,2]
                    else:
                        gradient = deltas_neuron.iloc[0,2] * outs_neuron[i-1]
                else:
                    if i == 0:
                        gradient = deltas_neuron.iloc[0,2]
                    else:
                        gradient = deltas_neuron.iloc[0,2] * outs_neuron.iloc[i-1,2]
                
                # Add the gradients of each weight of the neurons to the final DataFrame
                gradient = pd.DataFrame([layer+1, neuron+1, i, gradient]).transpose()
                gradient.columns=['Layer','Neuron', 'Weight', 'Gradient']
                gradients = pd.concat([gradients, gradient])
                gradients[['Layer','Neuron', 'Weight']] = gradients[['Layer','Neuron', 'Weight']].astype('int64')

    return gradients

## Ajust Weights
def adjustWeights(neurons, weights, gradients, previous_gradients, eta, mu, version):
    newWeights = weights.copy()

    # If we are in the offline version, then we use all the gradients together
    if version == 'offline':
        gradients_offline = gradients[0].copy()
        if len(previous_gradients) != 0:
            previous_gradients_offline = previous_gradients[0].copy()
        
        for gradient_pattern in range(len(gradients)):
            if gradient_pattern != 0:
                gradients_offline['Gradient'] += gradients[gradient_pattern]['Gradient']
                if len(previous_gradients) != 0:
                    previous_gradients_offline['Gradient'] += previous_gradients[gradient_pattern]['Gradient']

    # Calculate the new weights
    row = 0
    for layer in range(len(neurons)):
        for neuron in range(neurons[layer]):
            weight_neuron = weights[weights['Layer'] == layer+1]
            weight_neuron = weight_neuron[weight_neuron['Neuron'] == neuron+1]

            if version == 'online':
                gradient_neuron = gradients[gradients['Layer'] == layer+1]
                gradient_neuron = gradient_neuron[gradient_neuron['Neuron'] == neuron+1]

                if not previous_gradients.empty:
                    previous_gradient_neuron = previous_gradients[previous_gradients['Layer'] == layer+1]
                    previous_gradient_neuron = previous_gradient_neuron[previous_gradient_neuron['Neuron'] == neuron+1]

                for i in range(gradient_neuron.shape[0]):
                    if previous_gradients.empty:
                        newWeights.iloc[row,2+i] = weight_neuron.iloc[0,2+i] - eta * gradient_neuron.iloc[i,3]
                    else:
                        newWeights.iloc[row,2+i] = weight_neuron.iloc[0,2+i] - eta * gradient_neuron.iloc[i,3] - mu * (eta * previous_gradient_neuron.iloc[i,3])
            elif version == 'offline':
                gradient_neuron = gradients_offline[gradients_offline['Layer'] == layer+1]
                gradient_neuron = gradient_neuron[gradient_neuron['Neuron'] == neuron+1]

                if len(previous_gradients) != 0:
                    previous_gradient_neuron = previous_gradients_offline[previous_gradients_offline['Layer'] == layer+1]
                    previous_gradient_neuron = previous_gradient_neuron[previous_gradient_neuron['Neuron'] == neuron+1]

                for i in range(gradient_neuron.shape[0]):
                    if len(previous_gradients) == 0:
                        newWeights.iloc[row,2+i] = weight_neuron.iloc[0,2+i] - eta * gradient_neuron.iloc[i,3]
                    else:
                        newWeights.iloc[row,2+i] = weight_neuron.iloc[0,2+i] - eta * gradient_neuron.iloc[i,3] - mu * (eta * previous_gradient_neuron.iloc[i,3])

            row += 1

    return newWeights

## Calculate Error
def calculateError(desire, outs):
    outs_final = outs[outs['Layer'] == outs['Layer'].max()]

    error = 0
    for i in range(outs_final.shape[0]):
        out_i = outs_final.iloc[i,2]
        desire_i = desire.iloc[i,0]

        error += np.power(out_i - desire_i, 2)

    error /= outs_final.shape[0]

    return error

## Parameters Check
def parametersCheck(data, desire, neurons, eta, mu, version, function, error_target, max_iterations, debug):
    if type(data) != pd.DataFrame:
        raise Exception("The variable 'data' must be a Pandas DataFrame.")

    if type(desire) != pd.DataFrame:
        raise Exception("The variable 'desire' must be a Pandas DataFrame.")

    if desire.shape[1] != neurons[-1]:
        raise Exception("The number of output neurons needs to be the same as the len of desire values.")

    if type(neurons) != list:
        raise Exception("The variable 'Neurons' must be a list.")

    if type(eta) != float and type(eta) != int:
        raise Exception("The variable 'ETA' must be a integer or float.")

    if type(mu) != float and type(mu) != int:
        raise Exception("The variable 'MU' must be a integer or float.")

    if version != 'online' and version != 'offline':
        raise Exception("The variable 'VERSION' must be 'online' or 'offline'.")

    if function != 'sigmoid' and function != 'hyperbolic':
        raise Exception("The variable 'FUNCTION' must be 'sigmoid' or 'hyperbolic'.")

    if type(error_target) != float and type(error_target) != int:
        raise Exception("The variable 'error_target' must be a integer or float.")

    if type(max_iterations) != int:
        raise Exception("The variable 'max_iterations' must be a integer.")

    if type(debug) != bool:
        raise Exception("The variable 'debug' must be a boolean variables.")

## Final Outputs
def getFinalOutputs(data, neurons, weights, function):
    outputs = pd.DataFrame()

    desires = []
    for i in range(neurons[-1]):
        d = 'D' + str(i+1)
        desires .append(d)

    for row in range(data.shape[0]):
        pattern = data.iloc[row,:]

        outs = propagateInputs(pattern, neurons, weights, function)
        outs = outs[outs['Layer'] == outs['Layer'].max()]
        outs = list(outs['Value'])

        outs = pd.DataFrame(outs)
        outputs = pd.concat([outputs, outs.transpose()])

    outputs.columns = desires
    outputs.reset_index(drop=True, inplace=True)

    return outputs

## Final Error
def getFinalError(desire, outs):
    error = 0

    for i in range(outs.shape[0]):       
        for j in range(outs.shape[1]):    
            desire_pattern = desire.iloc[i,j]
            out_pattern = outs.iloc[i,j]

            error += np.power(desire_pattern - out_pattern, 2)
        error /= outs.shape[1]
    error /= outs.shape[0]
    
    return error

# Neural Network

class ann_classification:
    data = None
    desire = None
    neurons = None
    eta = None
    mu = None
    version = 'online'
    function = 'sigmoid'
    error_target = 0.01
    max_iterations = 100
    debug = False

    def __init__(self, data, desire, neurons, eta, mu, version='online', function='online', error_target=0.01, max_iterations=100, debug=False):
        self.data = data
        self.desire = desire
        self.neurons = neurons
        self.eta = eta
        self.mu = mu
        self.version = version
        self.function = function
        self.error_target = error_target
        self.max_iterations = max_iterations
        self.debug = debug

    def train(self):
        parametersCheck(self.data, self.desire, self.neurons, self.eta, self.mu, self.version, self.function, self.error_target, self.max_iterations, self.debug)
    
        weights = generateWeights(self.data, self.neurons)

        if self.debug == True:
            print('INITIAL WEIGHTS:\n', weights)
        
        error = 1
        repetitions = 0
        previous_gradients = pd.DataFrame()
        
        while (error >= self.error_target) and (repetitions < self.max_iterations):
            if self.version == 'online':
                error = 0
                for row in range(self.data.shape[0]):
                    pattern = self.data.iloc[row,:]
                    desire_row = list(self.desire.iloc[row])

                    outs = propagateInputs(pattern, self.neurons, weights, self.function)
                    deltas = backpropagateError(self.neurons, desire_row, outs, weights, self.function)
                    gradients = accumulateChange(pattern, self.neurons, outs, deltas)
                    weights = adjustWeights(self.neurons, weights, gradients, previous_gradients, self.eta, self.mu, self.version)
                    previous_gradients = gradients

            elif self.version == 'offline':
                error = 0
                gradients_patterns = []
                for row in range(self.data.shape[0]):
                    pattern = self.data.iloc[row,:]
                    desire_row = list(self.desire.iloc[row])

                    outs = propagateInputs(pattern, self.neurons, weights, self.function)
                    deltas = backpropagateError(self.neurons, desire_row, outs, weights, self.function)
                    gradients = accumulateChange(pattern, self.neurons, outs, deltas)
                    gradients_patterns.append(gradients)
                    
                weights = adjustWeights(self.neurons, weights, gradients_patterns, previous_gradients, self.eta, self.mu, self.version)
                previous_gradients = gradients_patterns

            outputs = getFinalOutputs(self.data, self.neurons, weights, self.function)
            error = getFinalError(self.desire, outputs)

            if self.debug == True and repetitions == 0:
                print('\nINITIAL ERROR:', error)

            repetitions += 1

        if self.debug == True:
            outputs = getFinalOutputs(self.data, self.neurons, weights, self.function)
            print('FINAL ERROR:', getFinalError(self.desire, outputs))
            print('Total Iterations:', repetitions)
            print('Outputs:\n' + str(outputs))
        
        return weights
