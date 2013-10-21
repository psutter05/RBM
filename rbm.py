import numpy as np
import math
import random

class RBM:
  def __init__(self, visible, hidden, learning_rate = 0.1):
    self.visible = visible
    self.hidden = hidden
    self.learning_rate = learning_rate

    #initialize the weight with an additional row and column for the bias weights 
    self.weights = np.random.random((visible+1, hidden+1))


  def train(self, data, epochs = float('inf'), error = 0.0000001):
    epoch = 0
   

    data = np.array(data)
    num_examples = data.shape[0]
    #insert the bias neurons into the data
    data = np.insert(data, 0, 1, axis = 1)
    data = np.insert(data, 0, 1, axis = 0)
    while epoch < epochs:
      epoch = epoch + 1
      
      #Calculate the state of the hidden neurons given the visible neurons
      h_act1 = np.dot(data, self.weights)
      h_prob1 = self.__logistic(h_act1)
      h_state1 = h_prob1 > random.random()

      pos_ass = np.dot(data.T, h_state1)
      
      #Regenerate the visible neurons from the hidden ones
      v_act = np.dot(h_state1, self.weights.T)
      v_prob = self.__logistic(v_act)
      v_state = v_prob > random.random()

      v_state[0,:] = v_state[:,0] = 1
      
      #Again, calculate the hidden neurons from visible
      h_act2 = np.dot(v_state, self.weights)
      h_prob2 = self.__logistic(h_act2)
      h_state2 =  h_prob2 > np.random.random(h_prob2.shape)

      neg_ass = np.dot(v_state.T, h_state2.astype(int))

      diff = (self.learning_rate / num_examples) * (pos_ass - neg_ass)
      cur_error = np.sum( data - v_state ) ** 2

      self.weights = self.weights + diff
      
      print cur_error

      if cur_error < error:
        break

      

      
  def __logistic(self, x):
    return 1/(1 + np.exp(-x))
