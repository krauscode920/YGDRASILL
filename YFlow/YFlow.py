import numpy as np
import random

def _set_global_seed(seed=42):
    """
    Set the seed for all random number generators used in the library.
    This function is called internally and doesn't need to be called by the user.
    """
    np.random.seed(seed)
    random.seed(seed)


class Layer:
    # Constructor for the Layer class
    def __init__(self):
        # 'input' will store the data that is fed into the layer during the forward pass
        self.input = None
        # 'output' will store the result after the layer processes the input during the forward pass
        self.output = None

    # The 'forward' method is intended to be implemented by subclasses.
    # It defines how the layer processes the input data to produce an output.
    # 'input' parameter: the data fed into the layer (e.g., activations from the previous layer)
    def forward(self, input):
        # This method is not implemented in the base Layer class.
        # Subclasses must override it to provide specific functionality.
        raise NotImplementedError

    # The 'backward' method is also intended to be implemented by subclasses.
    # It calculates how the layer's parameters should be updated during backpropagation.
    # 'output_gradient' parameter: the gradient of the loss function with respect to the layer's output
    def backward(self, output_gradient):
        # This method is not implemented in the base Layer class.
        # Subclasses must override it to handle gradient calculations for training.
        raise NotImplementedError


class Dense(Layer): # Middle + Output Layer
    # Initialize the Dense layer.
    def __init__(self, input_size, output_size, regularization=None, reg_strength=0.01):            #Class Constructor
        # Call the constructor of the parent Layer class to initialize any inherited attributes.
        super().__init__()
        _set_global_seed()  # Set seed before initializing weights
        # Initialize weights with small random values, shaped according to the input and output sizes.
        # This helps break symmetry and ensures that neurons learn different features.
        self.weights = np.random.randn(input_size, output_size) * 0.01

        # Initialize biases to zero for each output neuron.
        self.bias = np.zeros((1, output_size))

        # Store regularization method and its strength (used to prevent overfitting).
        self.regularization = regularization
        self.reg_strength = reg_strength

    # Forward pass: calculate the output of the layer given the input.
    def forward(self, input):
        # Store the input for use in the backward pass.
        self.input = input

        # Calculate the output as the dot product of the input and weights, plus the bias.
        return np.dot(self.input, self.weights) + self.bias  # WX + B

    # Backward pass: compute gradients for backpropagation.
    def backward(self, output_gradient):
        # Calculate the gradient of the weights by taking the dot product of the input and the output gradient.
        weights_gradient = np.dot(self.input.T, output_gradient)

        # Calculate the gradient of the input for the previous layer by taking the dot product of the output gradient and the transposed weights.
        input_gradient = np.dot(output_gradient, self.weights.T)

        # Apply regularization if specified (L2 or L1).
        if self.regularization == 'l2':
            # For L2 regularization, add the gradient of the weights scaled by the regularization strength.
            weights_gradient += self.reg_strength * self.weights
        elif self.regularization == 'l1':
            # For L1 regularization, add the gradient of the weights based on the sign of the weights.
            weights_gradient += self.reg_strength * np.sign(self.weights)

        # Store the gradient of the weights for updating them during optimization.
        self.weights_gradient = weights_gradient

        # Calculate the gradient of the bias as the sum of the output gradient along the batch axis.
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # Return the gradient of the input to pass it to the previous layer in the network.
        return input_gradient



#Dropout Regulatization
class Dropout(Layer):
    def __init__(self,dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, mode):
        _set_global_seed()  # Set seed before initializing weights
        if mode == "training":
            # Generate a random mask with the same shape as input
            self.mask = (np.random.rand(*input.shape) > self.dropout_rate).astype(float)
            # Multiply the input by the mask
            output = input * self.mask
            # Scale the output by (1 / (1 - dropout_rate))
            output *= (1 / (1 - self.dropout_rate))
            return output
        elif mode == "inference":
            # Return the input unchanged during inference
            return input

    def backward(self, output_gradient):
        # Scale the output gradient by the dropout mask
        # This ensures that gradients for the dropped-out neurons are zero
        output_gradient *= self.mask

        # Return the modified output gradient
        return output_gradient


#BatchNorm Regulatization
class BatchNormalization(Layer):
    def __init__(self, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon  # Small constant for numerical stability
        self.momentum = momentum  # Momentum for running mean and variance
        self.gamma = 1.0  # Initialize gamma (scaling factor)
        self.beta = 0.0  # Initialize beta (shift factor)
        self.running_mean = 0.0  # Initialize running mean
        self.running_variance = 1.0  # Initialize running variance

    def forward(self, input_data, training=True):
        if training:
            # Compute batch mean and variance
            batch_mean = np.mean(input_data, axis=0)
            batch_variance = np.var(input_data, axis=0)

            # Normalize input data
            normalized_input = (input_data - batch_mean) / np.sqrt(batch_variance + self.epsilon)

            # Update running mean and variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * batch_variance
        else:
            # Use running mean and variance for normalization during inference
            normalized_input = (input_data - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)

        # Scale and shift the normalized input
        output = self.gamma * normalized_input + self.beta

        return output

    def update_parameters(self, gamma_gradient, beta_gradient, learning_rate):
        # Update gamma and beta based on the gradients
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient


# ReLU activation layer class, which inherits from the Layer base class
class ReLU(Layer):
    # Forward pass for the ReLU activation function
    def forward(self, input):
        self.input = input  # Store the input for use in the backward pass
        # Return the input with all negative values set to zero (ReLU)
        return np.maximum(0, self.input)

    # Backward pass for the ReLU activation function
    def backward(self, output_gradient):
        # Gradient is passed through unchanged where input > 0, otherwise set to 0
        return output_gradient * (self.input > 0)

# Sigmoid activation layer class, which also inherits from the Layer base class
class Sigmoid(Layer):
    # Forward pass for the Sigmoid activation function
    def forward(self, input):
        self.input = input  # Store the input for use in the backward pass
        # Compute the Sigmoid output: 1 / (1 + exp(-input))
        self.output = 1 / (1 + np.exp(-self.input))
        # Return the computed Sigmoid output
        return self.output

    # Backward pass for the Sigmoid activation function
    def backward(self, output_gradient):
        # Gradient calculation for the Sigmoid derivative:
        # output_gradient * sigmoid_output * (1 - sigmoid_output)
        return output_gradient * self.output * (1 - self.output)

# BinaryCrossEntropy loss class for binary classification problems
class BinaryCrossEntropy:
    # Static method to calculate the loss value
    @staticmethod
    def calculate(y_pred, y_true):
        # Compute the Binary Cross-Entropy loss using y_pred (predicted) and y_true (actual) values
        # Add a small value (1e-15) to prevent log(0)
        return -np.mean(y_true * np.log(y_pred + 1e-15) + (1 - y_true) * np.log(1 - y_pred + 1e-15))

    # Static method to compute the derivative of the loss for backpropagation
    @staticmethod
    def derivative(y_pred, y_true):
        # Derivative calculation for Binary Cross-Entropy loss with respect to y_pred
        # This is used in backpropagation to update the model parameters
        return (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-15)

# Stochastic Gradient Descent (SGD) optimizer class
class SGD:
    # Initialize the optimizer with a learning rate, default is 0.01
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    # Update the model parameters based on gradients
    def update(self, params, grads):
        # Loop through each parameter and its corresponding gradient
        for param, grad in zip(params, grads):
            # Update the parameter by subtracting the learning rate multiplied by the gradient
            param -= self.learning_rate * grad

# Adam optimizer class, which is an adaptive learning rate optimization algorithm

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # CHANGE: Using dictionaries instead of lists for m and v
        # This allows us to store momentum and velocity for parameters of different shapes
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        # Initialize momentum and velocity if this is the first update
        if not self.m:
            for param_name, param in params.items():
                # Create zero-filled arrays with the same shape as each parameter
                self.m[param_name] = np.zeros_like(param)
                self.v[param_name] = np.zeros_like(param)

        self.t += 1
        # Compute the bias-corrected learning rate
        lr_t = self.learning_rate * (np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t))

        # CHANGE: Iterate over dictionary items instead of using enumerate
        # This ensures we update each parameter correctly, regardless of its shape
        for param_name in params:
            grad = grads[param_name]
            # Update biased first moment estimate
            self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * grad
            # Update biased second raw moment estimate
            self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * np.square(grad)

            # CHANGE: Update the parameter in place
            # This avoids any broadcasting issues by ensuring element-wise operations
            params[param_name] -= lr_t * self.m[param_name] / (np.sqrt(self.v[param_name]) + self.epsilon)


# Model class represents a neural network model that consists of multiple layers, a loss function, and an optimizer
class Model:
    def __init__(self, seed=None):
        # Initialize an empty list to hold the layers of the model
        self.layers = []
        # Placeholder for the loss function
        self.loss = None
        # Placeholder for the optimizer
        self.optimizer = None

        self.seed = 42 if seed is None else seed
        _set_global_seed(self.seed)  # Set the seed when the model is created

    # Method to add a new layer to the model
    def add(self, layer):
        # Append the layer to the list of layers
        self.layers.append(layer)

    # Method to set the loss function and optimizer for the model
    def compile(self, loss, optimizer):
        # Assign the provided loss function to the model
        self.loss = loss
        # Assign the provided optimizer to the model
        self.optimizer = optimizer

    # Forward pass through the network to compute the output for given input data X
    def forward(self, X):
        output = X  # Start with the input data
        # Pass the input through each layer in sequence
        for layer in self.layers:
            output = layer.forward(output)  # Compute the output of the current layer
        # Return the final output after passing through all layers
        return output

    # Backward pass through the network to update gradients for all layers
    def backward(self, gradient):
        # Loop through the layers in reverse order (from output layer to input layer)
        for layer in reversed(self.layers):
            # Compute the gradient for the current layer based on the gradient from the previous layer
            gradient = layer.backward(gradient)

    # Method to train the model using the provided training data and parameters

    def train(self, X, y, epochs, batch_size):
        for epoch in range(epochs):
            epoch_loss = 0
            # Shuffle the data for each epoch
            permutation = np.random.permutation(len(X))
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, len(X), batch_size):
                # Get batch
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)
                # Compute loss
                loss = self.loss.calculate(y_pred, y_batch)
                epoch_loss += loss

                # Backward pass
                grad = self.loss.derivative(y_pred, y_batch)
                self.backward(grad)

                # CHANGE: Collect parameters and gradients as dictionaries
                # This allows us to handle parameters of different shapes separately
                params = {}
                grads = {}
                for j, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        params[f'weights_{j}'] = layer.weights
                        grads[f'weights_{j}'] = layer.weights_gradient
                    if hasattr(layer, 'bias'):
                        params[f'bias_{j}'] = layer.bias
                        grads[f'bias_{j}'] = layer.bias_gradient

                # CHANGE: Update parameters using dictionaries
                # This ensures each parameter is updated correctly, regardless of its shape
                self.optimizer.update(params, grads)

                # CHANGE: Update layer parameters
                # After optimization, we need to update the actual layer parameters
                for j, layer in enumerate(self.layers):
                    if hasattr(layer, 'weights'):
                        layer.weights = params[f'weights_{j}']
                    if hasattr(layer, 'bias'):
                        layer.bias = params[f'bias_{j}']

            # Print the average loss for this epoch
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(X)}")


    # Method to make predictions using the trained model
    def predict(self, X):
        # Perform a forward pass with the input data to get the model's predictions
        return self.forward(X)

