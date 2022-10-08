#!/usr/bin/env python
# coding: utf-8

# In[127]:


from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np


# In[128]:


def get_circles():
    '''
    generate data
    :return: X: input data, y: given labels
    '''
    np.random.seed(0)
    X, y = datasets.make_circles(200, noise=0.20)
    # X, y = datasets.make_moons(200, noise=0.20)  # An array with 200 samples
    return X, y

def plot_decision_boundary(pred_func, X, y):
    '''
    plot the decision boundary
    :param pred_func: function used to predict the label
    :param X: input data
    :param y: given labels
    :return:
    '''
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.title('tanh')
    plt.show()


# In[129]:


ACTIVATIONS = {'tanh': np.tanh,'linear':lambda x: x,'relu':lambda x: np.maximum(x,0)}
dACTIVATIONS = {'tanh': lambda x: 1-np.square(np.tanh(x)),'linear':lambda x:1,'relu':lambda x:(x>0).astype(float)}


# In[130]:


class Layer(object):

    def __init__(self, input, output, actFun_type="tanh",reg_lambda = 0.01, seed=0):
        """
        The class for a single Layer in the net
        :param nn_dim: the dimension of the neural net
        :param prev_dim: the dimension of the previous layer
        :param actFun_type: the activation function we are using
        :param reg_lambda:
        """
        self.input = input
        self.output = output
        np.random.seed(seed)
        self.actFun_type = actFun_type
        self.diff_actFun = dACTIVATIONS[actFun_type]
        self.reg_lambda = reg_lambda
        # Initiliaze the weights and bias for this Layer
        self.W = np.random.randn(self.input, self.output) / np.sqrt(self.input)
        self.b = np.zeros((1, self.output))
        # Defining all my variables
        self.X = None
        self.z = None
        self.a = None
        self.dW = None
        self.db = None

    def actFun(self, z):
        '''
        actFun computes the activation functions
        :param z: net input
        :return: activations
        '''

        return ACTIVATIONS[self.actFun_type](z)

    def feedforward(self, X):
        """
        feedforward builds a layer of neurons and computes one probability
        :param X: input data
        :param actFun: activation function
        :return:
        """
        self.X = X
        self.z = np.dot(self.X, self.W) + self.b
        self.a = self.actFun(self.z)
        return self.a

    def backprop(self, d):
        """
        :param delta: The delta from previous layer without modification by activation.
        :return: The delta times the weights without activation
        """
        da = self.diff_actFun(self.z)
        delta = d * da
        self.dW = np.dot(self.X.T, delta) + self.reg_lambda * self.W
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)  # Which will be the new delta for the proceeding layers.

    def descent_update(self, epsilon):
        # Gradient descent parameter update
        self.W += -epsilon * self.dW
        self.b += -epsilon * self.db


# In[131]:


class FinalLayer(Layer):

    def __init__(self, input_size, output_size, seed=0):
        super(FinalLayer, self).__init__(input_size, output_size,"linear",
                                           reg_lambda=0.01, seed=0)
        self.p = None

    def actFun(self, z):
        '''
        actFun computes the sofmtax activation functions
        :param z: net input
        :return: activations
        '''
        exp_scores = np.exp(z - np.max(z))
        return exp_scores / (np.sum(exp_scores, axis=1, keepdims=True))

    def diff_actFun(self, z):
        '''
        diff_actFun computes the derivatives of the softmax functions wrt the layer input
        :param z: net input
        :return: NOTE THIS SHOULD NOT BE USED FOR THE FINAL LAYER.
        The only suitable func is softmax.
        '''

        raise ValueError("Softmax is only meant to be used as final layer")

    def backprop(self, y):
        """
        Implementing backpropagation to adjust the parameters after the feedforward step.
        :param x: input data
        :param y: given labels - Shape (samples,)
        """
        num_examples = len(self.X)
        delta = self.a
        delta[range(num_examples), y] -= 1

        dregW = (self.reg_lambda * self.W)
        self.dW = np.dot(self.X.T, delta) + dregW
        self.db = np.sum(delta, axis=0)

        return np.dot(delta, self.W.T)


# In[132]:


class DeepNeuralNetwork(NeuralNetwork):
    """
    This is a Deep Neural Net which takes in a Neural Net.
    """
    def __init__(self, layers):
        self.input_dim = layers[0].input
        self.output_dim = layers[-1].output
        self.hidden_layers = layers
        self.p = None

        # self.hidden_layers = []

    def feedforward(self, X):
        """
        The feedforward implementation for this thing
        :param X:
        :return:
        """
        assert self.input_dim == X.shape[1], "Input size %s but passed array with %s cells" \
                                            % (self.input_dim, X.shape[1])
        inter = X
        for layer in self.hidden_layers:
            inter = layer.feedforward(inter)
        self.p = inter
        return inter

    def backprop(self, y):
        """
        Backpropagation algorithm for this neural net
        :param y:
        :return: none
        """
        delta = y
        for layer in reversed(self.hidden_layers): # we can just delegate back to each composee
            delta = layer.backprop(delta)

    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        """
        fit_model uses backpropagation to train the network
        :param X: input data
        :param y: given labels
        :param num_passes: the number of times that the algorithm runs through the whole dataset
        :param print_loss: print the loss or not
        :return:
        """
        # Gradient descent.
        for i in range(0, num_passes):
            # Forward propagation
            self.feedforward(X)
            # Backpropagation
            self.backprop(y)
            for layer in self.hidden_layers:
                layer.descent_update(epsilon) # This function does the regularization and updating.
            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def calculate_loss(self, X, y):
        '''
        calculate_loss computes the loss for prediction
        :param X: input data
        :param y: given labels - Shape (samples,)
        :return: the loss for prediction
        '''
        num_examples = len(X)
        # Forward propagation
        self.feedforward(X)
        data_loss = np.sum(-np.log(self.p[np.arange(num_examples), y]))
        return (1.0 / num_examples) * data_loss

    def predict(self, X):
        '''
        predict infers the label of a given data point X
        :param X: input data
        :return: label inferred
        '''
        self.feedforward(X)
        return np.argmax(self.p, axis=1)


# In[ ]:


def main():
    # # generate and visualize Make-Moons dataset
    # X, y = generate_data()
    X, y = get_circles()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    sizes = [X.shape[1], 10, 10, 6, 4, 3, 3, 3, 3, 3]
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(Layer(sizes[i], sizes[i + 1],"tanh"))
    layers.append(FinalLayer(sizes[-1], 2)) # have to add the last layer!
    model = DeepNeuralNetwork(layers)

    model.fit_model(X, y, epsilon=0.001, num_passes=50000)
    model.visualize_decision_boundary(X, y)


if __name__ == "__main__":
       main()


# In[ ]:




