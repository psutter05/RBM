import sys

sys.path.append('/Users/freefrag/RBM/')
import numpy as np
import Image as pil
from rbm import RBM
import os

#Shortname for Smiley Face Generator!
class SFG:
  
  def __init__(self):
    self.image_width = self.image_height = 25
    self.visible_units = self.image_width * self.image_height
    self.hidden_units = 400
    self.rbm = RBM(self.visible_units, self.hidden_units, 0.05)

  #assumes there are only training images in the training_folder
  def train(self, training_folder, epochs = 200):
    data = []
    for training_image in os.listdir(training_folder):
      image = pil.open(training_folder + '/' + training_image)
      image = self.array_for_image(image)
      data.append(image)

    self.rbm.train(data, epochs, 4)
  
  #takes a pil Image and returns an arary of 1s and 0s
  def array_for_image(self, image):
    arr = np.array(image.convert("L")).flatten() > (255 / 2)
    return arr.astype(np.uint8)


  def regen_image(self, image, samples):
    data = self.array_for_image(image)
    (v, _) = self.rbm.regenerate([data],samples)
    return self.image_for_array(v[0])

  def image_for_array(self, array):
    img_array = []
    for row in range(0, self.image_height):
      img_array.append(array[row * self.image_width : (row+1) * self.image_width])

    img_array = np.asarray(img_array, np.uint8) * 255
    return pil.fromarray(img_array)

    
