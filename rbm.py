import numpy as np

class RBM:
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
  # If batch_size doesn't divide data then the last
  # data % batch_size elements are
  def train(self, data, epochs, batch_size):
    data = self.__prepare_data(data)
    vb = data.mean(axis = 0)
    vb = 1 / (1 - vb)
    self.visible_bias = np.log(vb)
    assert len(data.shape) == 2 # Data should come as an array of arrays

    (num_examples, data_size) = data.shape
    batches = num_examples / batch_size
    # This works as the result of the division is an integer
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, data_size))

    momentum = 0.3
    velocity = np.zeros(self.weights.shape)
    hg = np.zeros(self.hidden_bias.shape)
    vg = np.zeros(self.visible_bias.shape)
    for epoch in xrange(0, epochs):
      total_gradient = np.zeros(self.weights.shape)
      total_error = 0
      i = 0
      for batch in data:
        i+=1
        gradient, vb, hb, error = self.run_batch(batch)
        total_gradient += gradient
        total_error += error
        hg += hb
        vg += vb
        velocity = momentum * velocity + gradient
        self.weights *= 0.9998
        self.weights += velocity * self.learning_rate
        self.hidden_bias += hb
        self.visible_bias += vb
        print "after batch {0}, error {1}".format(i, error)
      total_gradient /= float(batches)
      total_error /= float(batches)
      vg/=float(batches)
      hg/=float(batches)

      print "after epoch {0}, average error: {1}".format(epoch, error)


      if epoch >= 1:
        momentum = 0.5
      if epoch >= 2:
        momentum = 0.7
      if epoch >= 6:
        momentum = 0.9


  def run_batch(self, v_prob1):
    # Compute the hidden states activated by our input
    #h_state1 = self.regenerate_hidden(v_state1)
    h_act1 = np.dot(v_prob1, self.weights) + self.hidden_bias
    h_prob1 = self.__logistic(h_act1)
    h_state1 = ( h_prob1 > np.random.random(h_prob1.shape) ).astype(np.int)

    pos_ass = self.__compute_associations(v_prob1, h_state1)

    v_act2 = np.dot(h_state1, self.weights.T) + self.visible_bias
    v_prob2 = self.__logistic(v_act2)
    h_act2 = np.dot(v_prob2, self.weights) + self.hidden_bias
    h_prob2 = self.__logistic(h_act2)

    # Compute the network's regeneration of the input
    #v_state2 = self.regenerate_visible(h_state1)
    #Compute the hidden neurons triggered by the regenerated input
    #h_state2 = self.regenerate_hidden(v_state2)

    neg_ass = self.__compute_associations(v_prob2, h_prob2)

    batch_size = v_prob1.shape[0]
    diff = pos_ass - neg_ass
    weight_gradient =  diff / float(batch_size)
    error = np.square(v_prob1 - v_prob2).sum() / float(batch_size)

    vbias_gradient = (v_prob1 - v_prob2).mean(axis=0) * self.learning_rate
    hbias_gradient = (h_prob1 - h_prob2).mean(axis=0) * self.learning_rate
    #hbias_gradient = 0.01 - (h_state1).mean(axis=0)

    return weight_gradient,vbias_gradient, hbias_gradient, error

  #Performs gibbs sampling on the data
  #Returns a tuple (Visible State, Hidden State)
  def regenerate(self, data, samples = 20):
    v_state = data
    for sample in range(0, samples):
      h_state, _ = self.regenerate_hidden(v_state)
      vs, v_state = self.regenerate_visible(h_state)

    return (vs, h_state)

  # Given the visible neurons this function regenerates the hidden neurons
  # Assumes that the first value in each row is a bias
  def regenerate_hidden(self, visible):
      h_act = np.dot(visible, self.weights)
      h_act += self.hidden_bias
      h_prob = self.__logistic(h_act)
      h_state = (h_prob > np.random.random(h_prob.shape)).astype(np.int)
      return h_state, h_prob

  # Given the hidden neurons this function regenerates the visible neurons
  # Assumes that the first value in each row is a bias
  def regenerate_visible(self, hidden):
      v_act = np.dot(hidden, self.weights.T)
      v_act += self.visible_bias
      v_prob = self.__logistic(v_act)
      v_state = (v_prob > np.random.random(v_prob.shape)).astype(np.int)
      return v_state, v_prob

  #Returns the neuron associations
  def __compute_associations(self, visible, hidden):
    return np.dot(visible.T, hidden)

  #Transforms the 2D array to a numpy matrix and add an additional row and column for the bias neurons
  def __prepare_data(self, data):
    data = np.array(data)
    return data

  def __logistic(self, x):
    return 1/(1 + np.exp(-x))

#  def train(self, data, epochs, classes):
#    data = self.__prepare_data(data)
#    if classes < 1:
#      self.train2(data, epochs)
#      return
#
#    num_examples = data.shape[0]
#    batch_size = num_examples / classes
#
#    for x in range(0, batch_size):
#      batch = []
#      for i in range(0,classes):
#        batch.append(data[batch_size * i + x])
#      self.train2(np.array(batch),epochs)
#
#  #Trains the machine on the samples for the given number of epochs
#  def train2(self, data, epochs = 100):
#    num_examples = data.shape[0]
#    for epoch in range(0,epochs):
#      (v_state, h_state) = self.regenerate(data, 1, False)
#      #Compute the associations between the data and the hidden neurons it set
#      pos_ass = self.__compute_associations(data, h_state)
#
#      (_, h_state) = self.regenerate(v_state, 1, False)
#      #Compute the associations between the regenerated data and the hidden neurons it set
#      neg_ass = self.__compute_associations(v_state, h_state)
#
#      diff = (self.learning_rate / num_examples) * (pos_ass - neg_ass)
#      cur_error = np.sum( data - v_state ) ** 2
#
#      self.weights = self.weights + diff
