from data import MNIST
from dbn import DBN
import numpy as np

d = DBN([28*28, 500,500, 2000], 10, 0.1)
images = MNIST.load_images('images')
labels = MNIST.load_labels('labels')

img = images[0:60000]
lbl = labels[0:60000]

tst_img = MNIST.load_images('test_images')
tst_lbl = MNIST.load_labels('test_labels')

d.pre_train(img,5,50)
for i in xrange(0, 100):
  d.train_labels(img, lbl, 50, 50)
  tst_class = d.classify(tst_img,10)
  print 'Error over test data: {0}'.format(1 - (tst_class*tst_lbl).mean() * 10)

#print d.sample(img[0:1],0)
#print d.sample(img[1:2],0)
#print 'layer 2'
#print d.sample(img[0:1],1)
#print d.sample(img[1:2],2)
#print 'layer 2'
#print d.sample(img[0:1],2)
#print d.sample(img[1:2],2)
#print 'layer 3'
#print d.sample(img[0:1],3)[0]
#print d.sample(img[1:2],3)[0]
#print 'labels'
print np.around(d.classify(img, 20), 2)[0:20]
print lbl[0:20]

tst_class = d.classify(tst_img,20)
err_test = 1 - ((tst_class * tst_lbl).mean() * 10)

print "TEST"
print np.around(tst_class)[0:10]
print tst_lbl[0:10]
print err_test

