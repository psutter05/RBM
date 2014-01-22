import numpy as np
from rbm import RBM

# The Deep Belief Network class
class DBN:

  # The DBN Constructor
  # topology_layer_sizes - topology of the DBN in a list
  # number_labels - The number of classifying labels
  # learning_rate - The learning rate
  def __init__(self, topology_layer_sizes, number_labels, learning_rate = 0.1):
    # Sets upn the parameters
    self._rbms = []
    self.learning_rate = learning_rate
    self.number_labels = number_labels
    self.number_inputs = topology_layer_sizes[0]
    self.number_layers = len(topology_layer_sizes)-1

    # Creates the RBM layers
    for i in xrange(0, self.number_layers):
      self._rbms.append(RBM(topology_layer_sizes[i], topology_layer_sizes[i+1], learning_rate))

    # The weights for the classifying labels
    self._label_weights = \
      np.random.normal(0,0.01,(topology_layer_sizes[self.number_layers], self.number_labels))

  # Performs unsupervised training on the network
  # Each RBM learns the topology_layer_sizesuration of the hidden layer below it
  # data - The input data
  # epochs - The number of epochs to pre train for
  # batch_size - The size of a batch of data
  def pre_train(self, data, epochs = 5, batch_size = 250):
    samples = data
    # Loops through each layer of the DBN
    for cur in xrange(0,self.number_layers):
      # Trains each layer on the output of the previous layer
      print 'pre-training layer {0} of the DBN'.format(cur)
      current_rbm = self._rbms[cur]
      current_rbm.train(samples, epochs, batch_size)
      (_, samples) = current_rbm.regenerate(samples, 1)

  # Performs one step of backpropagation on the very top level in order to learn the label weights
  # Uses the steepest-descent method of learning
  # inputs - The data
  # target_labels - The associated labels
  # epochs - The epochs to train on each label
  # batch_size - The size of each batch
  def train_labels(self, inputs, target_labels, epochs = 50, batch_size = 250):
    # Sorts the data into an appropriate format
    data = np.array(zip(inputs, target_labels))
    (num_examples, data_size) = data.shape
    batches = num_examples / batch_size
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, data_size))

    #Initialize the velocities
    # Uses the momentum to train the weights
    velocities, _, _ = self.compute_backprop_gradients([data[0][0][0]], [data[0][0][1]])
    momentum = 0.5
    for epoch in xrange(0, epochs):
      epoch_error = 0
      if epoch >= 2:
        momentum = 0.9
      for batch in data:
        (inputs, target_labels) = zip(*batch)
        gradients, biases, batch_error = self.compute_backprop_gradients(inputs, target_labels)
        epoch_error += batch_error

        velocities = [x + momentum * y for (x,y) in zip(gradients, velocities)]
        i = 0
        for rbm in self._rbms:
          rbm.weights += velocities[i] * self.learning_rate / batch_size
          rbm.hidden_bias += biases[i] * self.learning_rate
          i = i+1

        self._label_weights += velocities[i] * self.learning_rate / batch_size

      print 'after epoch {0}, error: {1}'.format(epoch, epoch_error / batches)

  # Computes the back propogation gradients
  # inputs - The input data
  # target_labels - The associated labels
  def compute_backprop_gradients(self, inputs, target_labels):
    data = [np.array(inputs)]
    #Calculate the inputs to each layer of the RBM
    for rbm in self._rbms:
      _, inputs = rbm.regenerate_hidden(inputs)
      data.append(inputs)

    data.reverse()

    label_act = np.dot(data[0], self._label_weights)
    label_probs = self.softmax(label_act)

    error = target_labels - label_probs
    error_gradient = np.dot(data[0].T, error)

    gradients = [error_gradient]
    biases = []
    # Now start propagating the error downwards
    error = np.dot(error, self._label_weights.T)
    i = 0
    for rbm in self._rbms[::-1]:
      error = error * (1-data[i]) * data[i]
      error_gradient = np.dot(data[i+1].T, error)
      error = np.dot(error,rbm.weights.T)
      gradients.append(error_gradient)
      biases.append(error_gradient.mean(axis=0))
      i = i + 1

    mean_class_error = 1 - (target_labels * label_probs).mean() * self.number_labels
    return gradients[::-1], biases[::-1], mean_class_error

  # Classifies input data using the DBN
  # data - The data to classify
  # samples - The number of Gibbs sampling
  def classify(self, data, samples = 1):
    data = self.sample(data, self.number_layers -1, samples)

    if (self.number_layers != 0):
      associative_rbm = self._rbms[self.number_layers-1]
      (_,last_coding) = associative_rbm.regenerate(data, samples)
      activations = np.dot(last_coding, self._label_weights)
    else:
      activations = np.dot(data, self._label_weights)

    return self.softmax(activations)


  # Samples from the given layer of the network.
  # data - The input data to sample
  # layer - The layer to sample up to
  # samples - The number of Gibbs sampling to do
  def sample(self, data, layer, samples = 1):
    for rbm in self._rbms[0:layer]:
      (_, data) = rbm.regenerate(data, samples)
    return data

  # Returns the softmax probabilities of each neuron being on
  # data - The input data to the softmax classifier
  def softmax(self,data):
    activations = np.exp(data)
    activations = activations / activations.sum(axis = 1)[:,np.newaxis]
    return activations

  # Gets the topology of the DBN
  def get_topology(self):
    topology = []
    topology.append(self.number_inputs)
    for i in xrange(0, self.number_layers):
      topology.append(self._rbms[i].hidden)
    topology.append(self.number_labels)
    return topology
