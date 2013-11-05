import Image as pil
import os
import sys
import numpy
import random

#Takes an image in array form and returns it with added noise
def add_noise(image, threshold):
  noisy_image = image.copy()
  for i, row in enumerate(noisy_image):
    for j, pixel in enumerate(row):
      #If a pixel is unlucky enough, flip it's value
      if random.random() > threshold:
        noisy_image[i][j] = 255 - pixel

  return noisy_image

def make_noisy_clones(base_file, clone_prefix, copies, threshold = 0.97):
  #open the file and convert it into a monochromatic format
  base_image = pil.open(base_file).convert("L")
  image_array = numpy.array(base_image)

  for x in range(copies):
    clone_array = add_noise(image_array, threshold)
    clone_image = pil.fromarray(clone_array)

    clone_path = clone_prefix + str(x+1) + '.png'
    clone_image.save(clone_path)

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print "USAGE: python noisy.py base_folder result_folder"
    sys.exit(1)

  base_folder = sys.argv[1]
  result_folder = sys.argv[2]

  for base_image in os.listdir(base_folder):
    image_name = os.path.splitext(base_image)[0]
    make_noisy_clones(base_folder + '/' + base_image, result_folder + '/' + image_name, 100)


