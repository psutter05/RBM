import numpy as np
from rbm import RBM

class DBN:
  def __init__(self, config, number_labels, learning_rate = 0.1):
    self._rbms = []
    self.learning_rate = learning_rate
    self.number_labels = number_labels
    self.number_layers = len(config)-1 

    for i in xrange(0, self.number_layers):
      self._rbms.append(RBM(config[i], config[i+1],0.1))

    self._label_weights = \
      np.random.normal(0,0.01,(config[self.number_layers], self.number_labels))

  # Performs unsupervised training on the network
  # Each RBM learns the configuration of the hidden layer below it
  def pre_train(self, data, epochs = 5):
    samples = data
    for cur in xrange(0,len(self._rbms)):
      print 'pre-training layer {0} of the DBN'.format(cur)
      current_rbm = self._rbms[cur]
      current_rbm.train(samples, epochs, 250)
      (_, samples) = current_rbm.regenerate(samples, 1)

  # Performs one step of backpropagation on the very top level in order to learn the label weights
  # Uses the steepest-descent method of learning
  def step_train_labels(self, inputs, target_labels, batch_size = 40, learning_rate = 0.1):

    data = np.array(zip(inputs, target_labels))

    (num_examples, data_size) = data.shape
    batches = num_examples / batch_size
    data = data[0:batch_size * batches]
    data = data.reshape((batches, batch_size, data_size))

    error = 0
    for batch in data:
      (inputs, target_labels) = zip(*batch)
      num_examples = len(data)
      top_state = self.sample(inputs, self.number_layers)
      label_act = np.dot(top_state, self._label_weights)
      #Add an extra 'dimension' to the label activations, so that the division works
      label_probs = self.softmax(label_act)

      act_der = top_state.mean(axis = 0)
      act_der = act_der * (1 - act_der)
      label_errors = target_labels - label_probs
      error_gradient = np.dot(top_state.T, label_errors)
      self._label_weights += learning_rate / num_examples * error_gradient
      error += (1 - (target_labels * label_probs).mean() * 10)

    print error / batches
    
  def classify(self, data, samples):
    data = self.sample(data, self.number_layers -1)
    associative_rbm = self._rbms[self.number_layers-1]

    (_,last_coding) = associative_rbm.regenerate(data, 1)
    activations = np.dot(last_coding, self._label_weights)
    return self.softmax(activations)


  # Samples from the given layer of the network.
  def sample(self, data, layer, samples = 1):
    for rbm in self._rbms[0:layer]:
      (_, data) = rbm.regenerate(data, samples) 
    return data

  # Returns the softmax probabilities of each neuron being on  
  def softmax(self,data):
    activations = np.exp(data)
    activations = activations / activations.sum(axis = 1)[:,np.newaxis]
    return activations
