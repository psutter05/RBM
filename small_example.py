from data import MNIST
from dbn import DBN
import numpy as np

d = DBN([28*28, 500, 2000], 10,0.1)
images = MNIST.load_images('test_images')
labels = MNIST.load_labels('test_labels')

img = images[0:100]
lbl = labels[0:100]

tst_img = images[9000:95000]
tst_lbl = labels[9000:95000]

d.pre_train(img,30,50)
for i in xrange(0, 100):
  d.train_labels(img, lbl, 3, 50)
  tst_class = d.classify(tst_img,20)
  print 'Error kurwa: {0}'.format( 1 - ((tst_class * tst_lbl).mean()) * 10)

#print d.sample(img[0:1],0) 
#print d.sample(img[1:2],0) 
#print 'layer 2'
#print d.sample(img[0:1],1) 
#print d.sample(img[1:2],1) 
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
err_test = 1 - ((tst_class * tst_lbl).mean()) * 10

print "TEST"
print np.around(tst_class)[0:10]
print tst_lbl[0:10]
print err_test

