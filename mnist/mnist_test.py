#import tensorflow lib
import tensorflow as tf

#load data set (mnist for example)
from tensorflow.examples.tutorials.mnist import input_data

#read data from downloaded files, the parameter "one_hot=True" means the outputs are one-hot vectors. say as the form of [0,0,0,1,0,0,0,0,0,0]T
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#first we define the model

#x, as the input is a 2D-tensor (or matrix) with the size of n*784
#where n denotes the number of samples and 784 is the total size of one single sample (28*28)
x = tf.placeholder(tf.float32, [None, 784])
#w is the weight matrix
w = tf.Variable(tf.zeros([784,10]))
#b is the bias vector
b = tf.Variable(tf.zeros([10]))
#y is the output which interprets what number was writen in the picture
y = tf.nn.softmax(tf.matmul(x, w) + b)
#y_ is the grand truth
y_ = tf.placeholder(tf.float32, [None, 10])

#use cross entropy to evaluate the parameters, the second parameter of reduc_sum() means operations are done on the 2nd (index 1) dimension of y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

#use greadient descent to optimize the model, the parameter "0.5" is the learning rate.
#minimize(cross_entropy) refers to as our target function is defined by cross_entropy and we should minimize it.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#always remember to initialize variables before running the model
init = tf.initialize_all_variables()

#define a session to solve this
sess = tf.Session()
#run initialization
sess.run(init)

#loop for some amount of iterations, for example 1000 in this case
for i in range(1000):
	#set the batch size as 100
	batch_xs, batch_ys = mnist.train.next_batch(100)
	#train step by step
	sess.run(train_step, feed_dict = {x: batch_xs, y_: batch_ys})
	#logging
	if(i%10==0):
		#calculate the correct percentage of the model
		#argmax(y, 1) returns the index of the highest entry in tensor "y" on the 2nd axis
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		acc = sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels})
		print 'iter:%d\t%f' % (i, acc)

