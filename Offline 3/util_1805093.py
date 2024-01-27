import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import pickle
import torchvision.datasets as ds
from torchvision import transforms
from sklearn.model_selection import train_test_split

######################################################################
# LOSS

# def mse(y_true, y_pred):
#     return np.mean(np.power(y_true - y_pred, 2))

# def mse_prime(y_true, y_pred):
#     return 2 * (y_pred - y_true) / np.size(y_true)

# def binary_cross_entropy(y_true, y_pred):
#     return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

# def binary_cross_entropy_prime(y_true, y_pred):
#     return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

def cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return np.mean(-np.sum(y_true * np.log(y_pred)))

def cross_entropy_prime(y_true, y_pred):
    return np.subtract(y_pred, y_true) / y_true.shape[0]

######################################################################
# LAYERS

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input, is_training):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate, is_training):
        # TODO: update parameters and return input gradient
        pass
    
    def cleanup(self):
        self.input = None
        self.output = None
        
class Adam:
    def __init__(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None
        self.t = 0

    def minimize(self, weights, bias, weights_gradient, bias_gradient, learning_rate):
        if self.m_weights is None:
            self.m_weights = np.zeros_like(weights)
            self.v_weights = np.zeros_like(weights)
            self.m_bias = np.zeros_like(bias)
            self.v_bias = np.zeros_like(bias)

        self.t += 1

        # Update biased first moment estimate
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * weights_gradient
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * bias_gradient

        # Update biased second moment estimate
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * weights_gradient**2
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * bias_gradient**2

        # Correct bias in first and second moment estimates
        m_weights_hat = self.m_weights / (1 - self.beta1**self.t)
        v_weights_hat = self.v_weights / (1 - self.beta2**self.t)
        m_bias_hat = self.m_bias / (1 - self.beta1**self.t)
        v_bias_hat = self.v_bias / (1 - self.beta2**self.t)

        # Update weights and bias
        weights -= learning_rate * m_weights_hat / (np.sqrt(v_weights_hat) + self.epsilon)
        bias -= learning_rate * m_bias_hat / (np.sqrt(v_bias_hat) + self.epsilon)

        return weights, bias
    
class Dense(Layer):
    def __init__(self, input_size, output_size):
        # xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
        self.bias = np.zeros((1, output_size))
        self.adam_optimizer = Adam()

    def forward(self, input, is_training):
        self.input = input
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient, learning_rate, is_training):
        weights_gradient = np.dot(self.input.T, output_gradient)
        input_gradient = np.dot(output_gradient, self.weights.T)

        self.weights, self.bias = self.adam_optimizer.minimize(
            self.weights, self.bias, weights_gradient, np.sum(output_gradient, axis=0, keepdims=True), learning_rate
        )

        return input_gradient

class Dropout(Layer):
    def __init__(self, rate=0.5):
        super().__init__()
        self.rate = rate
        self.mask = None

    def forward(self, x, is_training=True):
        if not is_training:
            return x
        self.mask = np.random.binomial(1, self.rate, size=x.shape) / self.rate
        return x * self.mask

    def backward(self, grad, learning_rate, is_training=True):
        if not is_training:
            return grad
        return grad * self.mask
    
######################################################################
# ACTIVATIONS
    
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input, is_training):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate, is_training):
        # print("output_gradient", output_gradient)
        # print("self.input", self.input)
        r = np.multiply(output_gradient, self.activation_prime(self.input))
        # print("r", r)
        return r

def relu(x):
    return np.maximum(0, x)

def relu_prime(x):
    return np.where(x > 0, 1, 0)
    # return x * (self.forward(x) > 0)
            
class ReLU(Activation):
    def __init__(self):
        super().__init__(relu, relu_prime)

# class Tanh(Activation):
#     def __init__(self):
#         def tanh(x):
#             return np.tanh(x)

#         def tanh_prime(x):
#             return 1 - np.tanh(x) ** 2

#         super().__init__(tanh, tanh_prime)

# class Sigmoid(Activation):
#     def __init__(self):
#         def sigmoid(x):
#             return 1 / (1 + np.exp(-x))

#         def sigmoid_prime(x):
#             s = sigmoid(x)
#             return s * (1 - s)

#         super().__init__(sigmoid, sigmoid_prime)
        
def softmax_activation(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

def softmax_activation_prime(x):
    # print("softmax_activation_prime")
    return np.ones_like(x)

class Softmax(Activation):
    def __init__(self):
        super().__init__(softmax_activation, softmax_activation_prime)
        
        
######################################################################
# NETWORK

def predict(network, input, is_training=False):
    output = input
    for layer in network:
        # print(layer, "forward")
        output = layer.forward(output, is_training)
    return output


def train(network, loss, loss_prime, x_train, y_train, x_val, y_val,
          epochs = 1000, learning_rate = 0.01, batch_size = 1024, verbose = True):
    
    training_loss = []
    validation_loss = []
    training_accuracy = []
    validation_accuracy = []
    training_f1 = []
    validation_f1 = []
    
    for e in range(epochs):
        
        # train on batches
        error = 0
        for i in range(0, len(x_train), batch_size):
            x = x_train[i:i + batch_size]
            y = y_train[i:i + batch_size]
            
            # forward
            output = predict(network, x, is_training=True)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                # print(layer, "backward")
                grad = layer.backward(grad, learning_rate, is_training=True)
                layer.cleanup()
                
        # training loss
        error /= len(x_train)
        training_loss.append(error)
        
        # training accuracy
        y_pred = np.argmax(predict(network, x_train), axis=1)
        y_true = np.argmax(y_train, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        training_accuracy.append(accuracy)
        
        # training f1
        f1 = f1_score(y_true, y_pred, average='macro')
        training_f1.append(f1)
    
        # validation loss
        output = predict(network, x_val)
        val_error = loss(y_val, output)
        val_error /= len(x_val)
        validation_loss.append(val_error)
        
        # validation accuracy
        y_pred = np.argmax(output, axis=1)
        y_true = np.argmax(y_val, axis=1)
        accuracy = accuracy_score(y_true, y_pred)
        validation_accuracy.append(accuracy)
        
        # validation macro f1
        f1 = f1_score(y_true, y_pred, average='macro')
        validation_f1.append(f1)      
        
        if verbose:
            print(f'epoch: {e + 1}/{epochs}, train_loss: {error:.4f}, val_loss: {val_error:.4f}, train_acc: {accuracy:.4f}, val_acc: {accuracy:.4f}, train_f1: {f1:.4f}, val_f1: {f1:.4f}')
            
    return training_loss, validation_loss, training_accuracy, validation_accuracy, training_f1, validation_f1
            
            
######################################################################
# UTILS

def to_ndarray(dataset):
    images = []
    labels = []
    
    for image, label in dataset:
        image = image.numpy().flatten()
        label = np.eye(26)[label - 1]
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)
