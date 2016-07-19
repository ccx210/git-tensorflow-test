#import tensorflow lib
import tensorflow as tf

#load data set (mnist for example)
from tensorflow.examples.tutorials.mnist import input_data

#define function for generating weights
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial)
#define function for generating biases
def bias_variable(shape):
	initial = tf.constant(0.1, shape = shape)
	return tf.Variable(initial)
#define function for convolution layers
def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')
#define function for max pooling layers
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


#read data from downloaded files, the parameter "one_hot=True" means the outputs are one-hot vectors. say as the form of [0,0,0,1,0,0,0,0,0,0]T
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#interactive session allows session start before the definition of computation graph
sess = tf.InteractiveSession()

#x, as the input is a 2D-tensor (or matrix) with the size of n*784
#where n denotes the number of samples and 784 is the total size of one single sample (28*28)
x = tf.placeholder(tf.float32, shape = [None, 784])
#y_ is the grand truth
y_ = tf.placeholder(tf.float32, shape = [None, 10])

#reshape x to a 4d tensor to apply the layer
x_image = tf.reshape(x, [-1, 28, 28, 1])

#define the model like this:
#input->conv1(convolution+bias+relu)->pool1->conv2->pool2->fc1->drop->fc2->softmax->output


w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)


w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


w_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)


keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#use cross entropy to evaluate the parameters, the second parameter of reduc_sum() means operations are done on the 2nd (index 1) dimension of y
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices = [1]))

#use ADAM method to optimize (minimize) the target function
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#calculate the correct percentage of the model
#argmax(y, 1) returns the index of the highest entry in tensor "y" on the 2nd axis
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#always remember to initialize variables before running the model
sess.run(tf.initialize_all_variables())

for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i % 100 ==0:
		train_accuracy = accuracy.eval(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0})
		print("step: %d, training accuracy %g" % (i, train_accuracy))
	train_step.run(feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict = {x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
