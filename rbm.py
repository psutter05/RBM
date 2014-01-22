import numpy as np

# The Restricted Boltzmann Machine class
class RBM:

  # The RBM Constructor
  # visible - The number of visible (input) nodes
  # hidden - The number of hidden (feature) nodes
  # The learning rate of the RBM, defaults to 0.1
  def __init__(self, visible, hidden, learning_rate = 0.1):
    self.visible = visible
    self.hidden = hidden
    self.learning_rate = learning_rate

    #initialize the weight with an additional row and column for the bias weights
    self.weights = np.random.normal(0,0.01,(visible, hidden))

    #set the hidden bias weights to 0
    self.hidden_bias = np.zeros(hidden)
    #set the visible bias weights to the approximate probability of a neuron being activated in the training data
    self.visible_bias = np.tile(1,visible)

  # Trains the RBM
  # data - The test data to train on. Should come as an array of arrays
  # epoch -  The number of epochs to train for (>0)
  # batch_size - The size of the batches to divide the data into (>0)
  def train(self, data, epochs, batch_size):

    # Prepares the data for training
    data = self.__prepare_data(data)

    # Calculates the starting visible bias (fixed for divide by zero if all 1)
    vb = data.mean(axis = 0)
    for i, v in enumerate(vb):
      if (v == 1):
        vb[i] = 0.99999
    vb = 1 / (1 - vb)
    self.visible_bias = np.log(vb)

    # Data should come as an array of arrays
    assert len(data.shape) == 2

    # Obtains the shape of the data and divides it into batches
    (num_examples, data_size) = data.shape
    batches = num_examples / batch_size
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, data_size))

    # Initialises the training values
    momentum = 0.3
    velocity = np.zeros(self.weights.shape)
    hg = np.zeros(self.hidden_bias.shape)
    vg = np.zeros(self.visible_bias.shape)

    # Trains through each epoch
    for epoch in xrange(0, epochs):

      # Initialises the gradients, errors and batch counter
      total_gradient = np.zeros(self.weights.shape)
      total_error = 0
      i = 0

      # Trains each batch of data
      for batch in data:
        i+=1
        # Calculates the training gradients and adjusts the values
        gradient, vb, hb, error = self.run_batch(batch)
        total_gradient += gradient
        total_error += error
        hg += hb
        vg += vb
        velocity = momentum * velocity + gradient
        # Adjusts the weights and the biases
        self.weights *= 0.9998
        self.weights += velocity * self.learning_rate
        self.hidden_bias += hb
        self.visible_bias += vb
        # Outputs training status
        print "after batch {0}, error {1}".format(i, error)
      # Calculates total learning gradients and error
      total_gradient /= float(batches)
      total_error /= float(batches)
      vg/=float(batches)
      hg/=float(batches)
      # Outputs epoch status
      print "after epoch {0}, average error: {1}".format(epoch, error)

      # Adjusts the momentum of the training
      if epoch >= 1:
        momentum = 0.5
      if epoch >= 2:
        momentum = 0.7
      if epoch >= 6:
        momentum = 0.9


  # Runs a batch through the RBM training function
  # v_prob1 - The input batch of data
  def run_batch(self, v_prob1):
    # Computes the hidden probabilites and activations
    h_act1 = np.dot(v_prob1, self.weights) + self.hidden_bias
    h_prob1 = self.__logistic(h_act1)
    h_state1 = ( h_prob1 > np.random.random(h_prob1.shape) ).astype(np.int)
    # Calculates the positive associations from this
    pos_ass = self.__compute_associations(v_prob1, h_state1)
    # Calculates the visible values and then the hidden values from this
    v_act2 = np.dot(h_state1, self.weights.T) + self.visible_bias
    v_prob2 = self.__logistic(v_act2)
    h_act2 = np.dot(v_prob2, self.weights) + self.hidden_bias
    h_prob2 = self.__logistic(h_act2)
    # Computes the negative associations from these values
    neg_ass = self.__compute_associations(v_prob2, h_prob2)

    # Gets the batch size and calculates the error and gradients
    batch_size = v_prob1.shape[0]
    diff = pos_ass - neg_ass
    weight_gradient =  diff / float(batch_size)
    error = np.square(v_prob1 - v_prob2).sum() / float(batch_size)

    # Adjusts the gradients accordingly
    vbias_gradient = (v_prob1 - v_prob2).mean(axis=0) * self.learning_rate
    hbias_gradient = (h_prob1 - h_prob2).mean(axis=0) * self.learning_rate
    return weight_gradient,vbias_gradient, hbias_gradient, error

  # Performs gibbs sampling on the data
  # data - Input data
  # samples - The number of times to sample the data
  def regenerate(self, data, samples = 20):
    # Initialises the visible state
    v_state = data
    # For each sample, regenerates the hidden nodes, then visible from this
    for sample in range(0, samples):
      h_state, _ = self.regenerate_hidden(v_state)
      vs, v_state = self.regenerate_visible(h_state)
    return (vs, h_state)

  # Given the visible neurons this function regenerates the hidden neurons
  # Assumes that the first value in each row is a bias
  # visible - The state of the visible nodes
  def regenerate_hidden(self, visible):
      # Computes the activations, then the probabilites then the states
      h_act = np.dot(visible, self.weights)
      h_act += self.hidden_bias
      h_prob = self.__logistic(h_act)
      h_state = (h_prob > np.random.random(h_prob.shape)).astype(np.int)
      return h_state, h_prob

  # Given the hidden neurons this function regenerates the visible neurons
  # Assumes that the first value in each row is a bias
  # hidden - The state of the hidden nodes
  def regenerate_visible(self, hidden):
      # Computes the activations, then the probabilites then the states
      v_act = np.dot(hidden, self.weights.T)
      v_act += self.visible_bias
      v_prob = self.__logistic(v_act)
      v_state = (v_prob > np.random.random(v_prob.shape)).astype(np.int)
      return v_state, v_prob

  # Returns the neuron associations
  # visible - The state of the visible nodes
  # hidden - The state of the hidden nodes
  def __compute_associations(self, visible, hidden):
    return np.dot(visible.T, hidden)

  # Transforms the 2D array to a numpy matrix and add an additional
  # row and column for the bias neurons
  # data - Raw input data
  def __prepare_data(self, data):
    data = np.array(data)
    return data

  # Computes the logistic function
  # x - The input parameter
  def __logistic(self, x):
    return 1/(1 + np.exp(-x))