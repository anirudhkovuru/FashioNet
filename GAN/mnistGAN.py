# Import tensorflow, numpy
import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt

# Importing our MNIST images
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

# Getting our MNIST images, will replace with our data and its parameters later
x_train = mnist.train.images[:55000, :]



# For showing dimensions of x_train

# x_train.shape

# Code for checking how any random image looks like

# randomNum = random.randint(0, 55000)
# image = x_train[randomNum].reshape([28, 28])
# plt.imshow(image, cmap=plt.get_cmap('gray_r'))
# plt.show()



# # # Discriminator

# A CNN classifier function that takes in an image (of size 28 x 28 x 1) as input. 
# The output will be a single scalar number activation that describes whether or not the input image is real or not.

# Helper functions for creating CNN's

# Kind of like a 4D convolution from signal analysis, input is x, filter is w, how much stride in each dimension 
# given by list of 4 ints, padding is SAME so that output size same as input size

def conv2d(x, w):
	return tf.nn.conv2d(input=x, filter=w, strides=[1, 1, 1, 1], padding='SAME')

# Applying an averaging filter on the input x by using filter of size 1x2x2x1

def avg_pool_2x2(x):
	return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def discriminator(x_image, reuse=False):
	with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
	
		# First convolution and pool layers
		# A single channel image with 8 filters applied on it

		# Make weights using a 5x5x1x8 tensor named d_wconv1 and filled with values whose mean is 0 and 
		# stddev(Standard deviation) is 0.02
		W_conv1 = tf.get_variable('d_wconv1', [5, 5, 1, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))

		# Make a tensor of length 8 to store the biases, all of it initialized to 0
		b_conv1 = tf.get_variable('d_bconv1', [8], initializer=tf.constant_initializer(0))

		# Apply relu activation function on the image and weights with the biases added 
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		
		# Average pool to send to next layer
		h_pool1 = avg_pool_2x2(h_conv1)



		# Second convolution and pool layers
		# An 8 channel image with 16 filters applied on it
		
		# Make weights using a 5x5x8x16 tensor named d_wconv2 and filled with values whose mean is 0 and 
		# stddev(Standard deviation) is 0.02
		W_conv2 = tf.get_variable('d_wconv2', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
		
		# Make a tensor of length 16 to store the biases, all of it initialized to 0
		b_conv2 = tf.get_variable('d_bconv2', [16], initializer=tf.constant_initializer(0))
		
		# Apply relu activation function on the previous layer and the weights with the biases added
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		
		# Average pool to send to next layer
		h_pool2 = avg_pool_2x2(h_conv2)



		# First fully connected layer
		
		# Make weights matrix from the 7*7*16 neurons to 32 neurons filled with values whose mean is 0 and 
		# stddev(Standard deviation) is 0.02
		W_fc1 = tf.get_variable('d_wfc1', [7*7*16, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
		
		# Make a tensor of length 32 to store the biases, all of it initialized to 0
		b_fc1 = tf.get_variable('d_bfc1', [32], initializer=tf.constant_initializer(0))
		
		# Reshape the output of the first two layers to fit into the next layer
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*16])
		
		# Normal neural network transfer by multiplying weights and then adding biases
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



		# Second fully connected layer and Final layer
		
		# Make weights matrix from 32 neurons to 1 neuron filled with values whose mean is 0 and
		# stddev(Standard deviation) is 0.02
		W_fc2 = tf.get_variable('d_wfc2', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
		
		# Make tensor of length 1 to store the biases, all of it initialized to 0
		b_fc2 = tf.get_variable('d_bfc2', [1], initializer=tf.constant_initializer(0))
		
		# Final transfer by multiplying weights and then adding biases
		y_conv = (tf.matmul(h_fc1, W_fc2) + b_fc2)
		return y_conv

# # # Generator

# A generator model introduced in the DCGAN paper (link: https://arxiv.org/pdf/1511.06434v2.pdf). 
# The generator is a kind of reverse ConvNet.
# With CNNs, the goal is to transform a 2 or 3 dimensional matrix of pixel values into a single probability.
# A generator, however, seeks to take a d-dimensional noise vector and upsample it to become a 28 x 28 image.
# This upsampling is done through a convolutional transpose (or deconvolution) layer.
# ReLUs and Batch Norm are then used to stabilize the outputs of each layer.

# The structure of the generator is very similar to that of the discriminator, 
# except we're calling the convolution transpose method, instead of the conv2d one.

# The conv transpose + relu + batch norm pipeline is repeated 4 times so that the 
# output volume grows larger and larger until a 28 x 28 x 1 image is formed.  

def generator(z, batch_size, z_dim, reuse=False):
	if (reuse):
		tf.get_variable_scope().reuse_variables()
	
	# Number of filters of first layer	
	g_dim = 64 
	# Color dimension of the output
	c_dim = 1
	# Output size of the image
	s = 28
	# For gradual increase in size of the image
	s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

	# Turns the noise into a batch_size x 2 x 2 x 25 tensor. Batch size number of 2 x 2 images with 25 channels
	h0 = tf.reshape(z, [batch_size, s16+1, s16+1, 25])
	h0 = tf.nn.relu(h0)

	# First deconv layer
	output1_shape = [batch_size, s8, s8, g_dim*4]
	W_conv1 = tf.get_variable('g_wconv1', [5, 5, output1_shape[-1], int(h0.get_shape()[-1])],
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	b_conv1 = tf.get_variable('g_bconv1', [output1_shape[-1]], initializer=tf.constant_initializer(.1))
	H_conv1 = tf.nn.conv2d_transpose(h0, W_conv1, output_shape=output1_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv1
	H_conv1 = tf.contrib.layers.batch_norm(inputs = H_conv1, center=True, scale=True, is_training=True, scope="g_bn1")
	H_conv1 = tf.nn.relu(H_conv1)
	# Dimensions of H_conv1 = batch_size x 3 x 3 x 256

	# Second deconv layer
	output2_shape = [batch_size, s4-1, s4-1, g_dim*2]
	W_conv2 = tf.get_variable('g_wconv2', [5, 5, output2_shape[-1], int(H_conv1.get_shape()[-1])],
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	b_conv2 = tf.get_variable('g_bconv2', [output2_shape[-1]], initializer=tf.constant_initializer(.1))
	H_conv2 = tf.nn.conv2d_transpose(H_conv1, W_conv2, output_shape=output2_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv2
	H_conv2 = tf.contrib.layers.batch_norm(inputs = H_conv2, center=True, scale=True, is_training=True, scope="g_bn2")
	H_conv2 = tf.nn.relu(H_conv2)
	# Dimensions of H_conv2 = batch_size x 6 x 6 x 128

	# Third deconv layer
	output3_shape = [batch_size, s2-2, s2-2, g_dim*1]
	W_conv3 = tf.get_variable('g_wconv3', [5, 5, output3_shape[-1], int(H_conv2.get_shape()[-1])],
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	b_conv3 = tf.get_variable('g_bconv3', [output3_shape[-1]], initializer=tf.constant_initializer(.1))
	H_conv3 = tf.nn.conv2d_transpose(H_conv2, W_conv3, output_shape=output3_shape, strides=[1, 2, 2, 1], padding='SAME') + b_conv3
	H_conv3 = tf.contrib.layers.batch_norm(inputs = H_conv3, center=True, scale=True, is_training=True, scope="g_bn3")
	H_conv3 = tf.nn.relu(H_conv3)
	# Dimensions of H_conv3 = batch_size x 12 x 12 x 64

	# Fourth deconv layer
	output4_shape = [batch_size, s, s, c_dim]
	W_conv4 = tf.get_variable('g_wconv4', [5, 5, output4_shape[-1], int(H_conv3.get_shape()[-1])],
		initializer=tf.truncated_normal_initializer(stddev=0.1))
	b_conv4 = tf.get_variable('g_bconv4', [output4_shape[-1]], initializer=tf.constant_initializer(.1))
	H_conv4 = tf.nn.conv2d_transpose(H_conv3, W_conv4, output_shape=output4_shape, strides=[1, 2, 2, 1], padding='VALID') + b_conv4
	H_conv4 = tf.nn.tanh(H_conv4)
	# Dimensions of H_conv4 = batch_size x 28 x 28 x 1

	return H_conv4

# Code to check image generated by untrained generator

# sess = tf.Session()
z_dimensions = 100
# z_test_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])


# Now, we create a variable (sample_image) that holds the output of the generator,
# and also initialize the random noise vector that we’ll use as input.
# The np.random.normal function has three arguments. 
# The first and second define the range of the output distribution we want (between -1 and 1 in our case), 
# and the third defines the the shape of the vector (1 x 100).

# sample_image = generator(z_test_placeholder, 1, z_dimensions)
# test_z = np.random.normal(-1, 1, [1,z_dimensions])

# Next, we initialize all the variables, feed our test_z into the placeholder, and run the session. 
# The sess.run function has two arguments. The first is called the "fetches" argument. 
# It defines the value for you're interested in computing. 
# For example, in our case, we want to see what the output of the generator is. 
# If you look back at the last code snippet, the output of the generator function is stored in sample_image. 
# Therefore, we'll use sample_image for our first argument. The second argument is where we input our feed_dict. 
# This data structure is where we provide inputs to all of our placeholders. 
# In our example, we need to feed our test_z variable into the z placeholder we defined earlier. 

# sess.run(tf.global_variables_initializer())
# temp = (sess.run(sample_image, feed_dict={z_test_placeholder: test_z}))


# Finally, we can view the output through matplotlib. 

# my_i = temp.squeeze()
# plt.imshow(my_i, cmap='gray_r')
# plt.show()

# # # Training a GAN

batch_size = 16
# Resetting the tensorflow graph
tf.reset_default_graph()

sess = tf.Session()
# Placeholder for input images to the discriminator
x_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1])
# Placeholder for input noise vectors to the generator
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

# Dx will hold discriminator prediction probabilities for the real MNIST images
Dx = discriminator(x_placeholder)
# Gz will hold the generated images
Gz = generator(z_placeholder, batch_size, z_dimensions)
# Dg will hold discriminator prediction probabilities for the generated images
Dg = discriminator(Gz, reuse=True)

# # Loss calculation for the generator based on the label of 1 and the probabilities given by discriminator
# # for the generated image

# reduce_mean function to convert the vector returned by cross entropy function into a single scalar value
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# # Loss calculations for the discriminator based on realness or fakeness decided by 1 or 0

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
d_loss = d_loss_real + d_loss_fake

# # Defining our optimizers

# Splitting the discriminator and generator weights and biases for training
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Specifying our optimizers using Adam
with tf.variable_scope(tf.get_variable_scope()):
	print("reuse or not: {}".format(tf.get_variable_scope().reuse))
	assert tf.get_variable_scope().reuse == False, "Houston, we're fucked!!"
	trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
	trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list = g_vars)
print("exiting")

# Training the nets

print("Training Begin")
sess.run(tf.global_variables_initializer())
iterations = 3000
for i in range(iterations):
	z_batch = np.random.normal(-1, 1, size=[batch_size, z_dimensions])
	real_image_batch = mnist.train.next_batch(batch_size)
	real_image_batch = np.reshape(real_image_batch[0], [batch_size, 28, 28, 1])
	# Update the discriminator
	_, dLoss = sess.run([trainerD, d_loss], feed_dict={z_placeholder:z_batch, x_placeholder:real_image_batch})
	# Update the generator
	_, gLoss = sess.run([trainerG, g_loss], feed_dict={z_placeholder:z_batch})

print("Training End")

# # # GAN in action!

tf.get_variable_scope().reuse_variables()

sample_image = generator(z_placeholder, 1, z_dimensions)
z_batch = np.random.normal(-1, 1, size=[1, z_dimensions])
temp = (sess.run(sample_image, feed_dict={z_placeholder:z_batch}))
my_i = temp.squeeze()
plt.imshow(my_i, cmap='gray_r')
plt.show()
