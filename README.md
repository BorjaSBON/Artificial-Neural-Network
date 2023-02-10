# Artificial-Neural-Network

Implementation of a neural network to solve regression and classification problems. 

## Content

I have implemented two functions, one for the regression problem and another for the classifiction problem. These fucntions have different parameters that change the way that the neural networks work. The parameters are the following: 

- <b>Data</b>: Is the data that the fucntion will use to learn with <i>(DataFrame)</i>.
- <b>Desire</b>: Is the desire output that the function will try to approximate <i>(DataFrame)</i>.
- <b>Neurons</b>: Is the number of neurons that the neural network will have <i>(List)</i>. For example: [2, 1, 2] -> The first layer has 2 neurons, the second layer has only one neuron and the last layer has 2 neurons. The last layer must have the same number of neurons as number of desire outputs.
- <b>Eta</b>: Controls the steps taken to obtain the gradient vector <i>(Float)</i>.
- <b>Mu</b>: Controls the momentum <i>(Float)</i>.
- <b>Version</b>: Is the way that the function will calculate the errors and backpropagate <i>('Online' or 'Offline')</i>. 
- <b>Function</b>: Is the tyepe of function that the neural networks will use to propagate <i>('Sigmoid' or 'Hyperbolic')</i>.
- <b>Error Target</b>: When the error is inferior than the error target, the function will stop working <i>(Float)</i>.
- <b>Max Iterations</b>: When the iterations of the method are superior than the max iterations, the function will stop working<i>(Integer)</i>
- <b>Debug</b>: For each iteration, the user can receive some feedback <i>(Bool)</i>

To solve the problems, the functions have different methods to calculate the backpropagation and adjust the bias. These method are the following: 

- <b>Generate Weights</b>: This function generate random weights for each bias, it takes into account the number of inputs and the number of neurons for each layer. 
- <b>Propagate Inputs</b>: Calculate the outputs depeding in the type of function introduce.
- <b>Backpropagte Error</b>: Calculate the errors for each neuron backpropagating.
- <b>Accumulate Change</b>: Calculate the gradients for each bias of each neuron.
- <b>Adjust Weights</b>: Calculate the new weights.
