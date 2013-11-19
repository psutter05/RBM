import struct
import copy
import array
import numpy as np

class MNIST:
  @staticmethod
  def load_labels(file_path):
    data = IDX.load(file_path)
    data = np.array(data)
    assert len(data.shape) == 1 #labels are 1-dimensional

    class_vector_dict = {}
    for i in xrange(0, 10):
      #the ith class is represented as the ith unit vector
      class_vector_dict[i] = np.eye(1, 10, i).flatten()

    labels = map(class_vector_dict.get, data)
    return np.array(labels)

  @staticmethod
  def load_images(file_path):
    data = IDX.load(file_path)
    data = np.array(data)
    assert len(data.shape) == 3 #images are 3-dimensional

    num_examples = data.shape[0]
    pixels_per_image = data.shape[1] * data.shape[2]
    data = np.reshape(data, (num_examples, pixels_per_image))

    data = data/float(255) # normalise the input pixels, so that each takes a value from [0,1]

    data = data >= 0.5
    return data.astype(np.int)

class IDX:
  @classmethod
  def load(cls, file_path):
    f = open(file_path)
    (magic, data_code, dimensions) = struct.unpack('hBB', f.read(4))
    assert magic == 0 #The first two bytes are 0 

    data_format = cls.get_data_format(data_code)
    data_format_size = struct.calcsize(data_format)
    dimension_sizes = array.array('I',f.read(dimensions * 4))
    dimension_sizes.byteswap() # The IDX format is big endian, python is little endian

    def read_dimension(dimension):
      if dimension == (dimensions - 1): # dimension is 0-indexed, dimensions 1-indexed
        last_dim_bytes = dimension_sizes[dimension] * data_format_size
        result = array.array(data_format, f.read(last_dim_bytes))
        result.byteswap()
        return result
      
      result = []
      for x in xrange(0, dimension_sizes[dimension]):
        result.append(read_dimension(dimension+1))
      
      return result

    return read_dimension(0)
    
  @staticmethod
  def get_data_format(code):
    code_dict = {0x08 : 'B',
                 0x09 : 'b',
                 0x0B : 'H',
                 0x0C : 'I',
                 0x0D : 'f',
                 0x0E : 'd'}
    
    assert code in code_dict # We assume the code is known
    return code_dict[code]
