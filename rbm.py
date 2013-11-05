import numpy as np
import math

class RBM:
  def __init__(self, visible, hidden, learning_rate = 0.01):
    self.visible = visible
    self.hidden = hidden
    self.learning_rate = learning_rate

    #initialize the weight with an additional row and column for the bias weights 
    self.weights = np.random.normal(0,0.01,(visible+1, hidden+1))

    #set the hidden bias weights to 0
    self.weights[:,0] = 0

    #set the visible bias weights to the approximate probability of a neuron being activated in the training data
    self.weights[0,:] = 0.8

  def train(self, data, epochs, classes):
    data = np.array(data)
    if classes < 1:
      self.train2(data, epochs)
      return

    num_examples = data.shape[0]
    batch_size = num_examples / classes

    for x in range(0, batch_size):
      batch = []
      for i in range(0,classes):
        batch.append(data[batch_size * i + x])
      self.train2(batch,epochs)

  #Trains the machine on the samples for the given number of epochs
  def train2(self, data, epochs = 9999):
    data = np.array(data)
    num_examples = data.shape[0]
    for epoch in range(0,epochs):
      (v_state, h_state) = self.regenerate(data, 1)
      #Compute the associations between the data and the hidden neurons it set
      pos_ass = self.__compute_associations(data, h_state)

      (_, h_state) = self.regenerate(v_state, 1)
      #Compute the associations between the regenerated data and the hidden neurons it set
      neg_ass = self.__compute_associations(v_state, h_state)

      diff = (self.learning_rate / num_examples) * (pos_ass - neg_ass)
      cur_error = np.sum( data - v_state ) ** 2

      self.weights = self.weights + diff

  #Performs gibbs sampling on the data
  #Returns a tuple (Visible State, Hidden State)
  def regenerate(self, data, samples = 20):
    data = self.__prepare_data(data)

    v_state = data
    for sample in range(0, samples):
      #Calculate the state of the hidden neurons given the visible neurons
      h_act = np.dot(data, self.weights)
      h_prob = self.__logistic(h_act)
      h_state = (h_prob > np.random.random(h_prob.shape)).astype(int)
      h_state[:,0] = 1

      #Regenerate the visible neurons from the hidden ones
      v_act = np.dot(h_state, self.weights.T)
      v_prob = self.__logistic(v_act)
      v_state = (v_prob > np.random.random(v_prob.shape)).astype(int)

     # #Set the bias neurons
     # v_state[:,0] = 1

    #Delete the row containing bias neurons
    v_state = np.delete(v_state, 0, axis = 1)
    h_state = np.delete(h_state, 0, axis = 1)
    return (v_state, h_state)

  #Returns the neuron associations
  def __compute_associations(self, visible, hidden):
    visible = np.insert(visible, 0, 1, axis=1)
    hidden  = np.insert(hidden, 0, 1, axis=1)
    return np.dot(visible.T, hidden)

  #Transforms the 2D array to a numpy matrix and add an additional row and column for the bias neurons
  def __prepare_data(self, data):
    data = np.array(data)
    data = np.insert(data, 0, 1, axis = 1)
    return data

  def __logistic(self, x):
    return 1/(1 + np.exp(-x))
