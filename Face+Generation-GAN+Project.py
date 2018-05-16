
# coding: utf-8

# # Face Generation
# In this project, I'll use generative adversarial networks to generate new images of faces.
# ### The Data
# I've used two datasets in this project:
# - MNIST
# - CelebA
# 
# Since the celebA dataset is complex, I'll test the neural network on MNIST before CelebA.  Running the GANs on MNIST will allow to see how well the model trains, sooner.
# 

# In[1]:

#data_dir = './data'

# FloydHub - Used data ID "R5KrjnANiKVhLWAkpXhNBe"
data_dir = '/input'

import helper

helper.download_extract('mnist', data_dir)
helper.download_extract('celeba', data_dir)


# ## Exploring the Data
# ### MNIST
# The [MNIST](http://yann.lecun.com/exdb/mnist/) dataset contains images of handwritten digits. 

# In[2]:

show_n_images = 25

get_ipython().magic('matplotlib inline')
import os
from glob import glob
from matplotlib import pyplot

mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'mnist/*.jpg'))[:show_n_images], 28, 28, 'L')
pyplot.imshow(helper.images_square_grid(mnist_images, 'L'), cmap='gray')


# ### CelebA
# The [CelebFaces Attributes Dataset (CelebA)](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset contains over 200,000 celebrity images with annotations.  

# In[3]:

show_n_images = 25
#No annotations required since we'll be generating faces 
mnist_images = helper.get_batch(glob(os.path.join(data_dir, 'img_align_celeba/*.jpg'))[:show_n_images], 28, 28, 'RGB')
pyplot.imshow(helper.images_square_grid(mnist_images, 'RGB'))


# ## Preprocessing the Data
# The values of the MNIST and CelebA dataset will be in the range of -0.5 to 0.5 of 28x28 dimensional images.  The CelebA images will be cropped to remove parts of the image that don't include a face, then resized down to 28x28.
# 
# The MNIST images are black and white images with a single [color channel](https://en.wikipedia.org/wiki/Channel_(digital_image%29) while the CelebA images have [3 color channels (RGB color channel)](https://en.wikipedia.org/wiki/Channel_(digital_image%29#RGB_Images).
# ## Building the Neural Network
# I've build the components necessary to build a GANs by implementing the following functions below:
# - `model_inputs`
# - `discriminator`
# - `generator`
# - `model_loss`
# - `model_opt`
# - `train`
# 
# Note: Do check the version of Tensorflow and GPU access before starting.

# In[4]:

from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


# ### Input
# The function `model_inputs` creates the following TF Placeholders for the Neural Network:
# - Real input images placeholder with rank 4 using `image_width`, `image_height`, and `image_channels`.
# - Z input placeholder with rank 2 using `z_dim`.
# - Learning rate placeholder with rank 0.
# 
# The placeholders are returned in the following the tuple (tensor of real input images, tensor of z data)

# In[5]:

import problem_unittests as tests

def model_inputs(image_width, image_height, image_channels, z_dim):
    
    # TODO: Implement Function
    real_inputs = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='real_inputs')
    z_inputs = tf.placeholder(tf.float32, (None, z_dim), name='z_inputs')
    learning_rate= tf.placeholder(tf.float32,(None),name='learning_rate')

    return real_inputs,z_inputs, learning_rate

tests.test_model_inputs(model_inputs)


# ### The Discriminator lol
# The discriminator neural network discriminates on `images`.  This function reuses the variables in the neural network using [`tf.variable_scope`](https://www.tensorflow.org/api_docs/python/tf/variable_scope) with a scope name of "discriminator" to allow the variables to be reused.

# In[80]:

def discriminator(images, reuse=False):
    
    # TODO: Implement Function
    with tf.variable_scope('discriminator', reuse = reuse):
        #x1 = tf.contrib.layers.xavier_initializer_conv2d(images, filters=64, kernel_size=5, strides=2, padding='same')
        x1 = tf.layers.conv2d(images, filters=128, kernel_size=5, strides=2, padding='same')
        x1 = tf.maximum(0.2 * x1, x1)
        #x1 = tf.nn.dropout(x1, keep_prob=0.8)
        #print(x1.shape)
        # 14x14x64
         
        x2 = tf.layers.conv2d(x1, filters=256, kernel_size=5, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training= True)
        x2 = tf.maximum(0.2 * x2, x2)
        #x2 = tf.nn.dropout(x2, keep_prob=0.8)
        
        
        #print(x2.shape)
        # 6x6x128
         
        x3 = tf.layers.conv2d(x2, filters=512, kernel_size=5, strides=2, padding='valid')
        x3 = tf.layers.batch_normalization(x3, training= True)
        x3 = tf.maximum(0.2 * x3, x3)
        #x3 = tf.nn.dropout(x3, keep_prob=0.8)
        print(x3.shape)
        # 4x4x256
        
        #x4 = tf.layers.conv2d(x3, filters=512, kernel_size=5, strides=2, padding='same')
        #x4 = tf.layers.batch_normalization(x4, training= True)
        #x4 = tf.maximum(0.2 * x4, x4)
        #x4 = tf.nn.dropout(x4, keep_prob=0.8)
        #print(x4.shape)
        # 2x2x512
        
        flat = tf.reshape(x3, (-1, 4*4*512))
        logits = tf.layers.dense(flat, 1)
        out = tf.sigmoid(logits)
        return out, logits

    """
    def conv2d_layer(x):
        x = tf.layers.conv2d(x, int(x.get_shape()[3])*2, 5, strides =2, padding='same')
        x = tf.layers.batch_normalization(x, training=True)
        x = tf.maximum(0.2 * x, x)
        return x
    with tf.varaiable_scope('discriminator', reuse=reuse):
        x1 = tf.layers.conv2d(images, 64, 5, strides =2, padding='same')
        x1 = tf.maximum(0.2 * x1, x1)
        x2 = conv2d_layer(x1)
        x3 = conv2d_layer(x2)
        d1,d2,d3,d4 =list(x3.get_shape())
        x4 = tf.reshape(x3, (-1, int(d2)*int(d3)*int(d4)))
        logits = tf.layers.dense(x4,1)
        out =tf.sigmoid(logits)

    return out, logits
    """

tests.test_discriminator(discriminator, tf)


# ### Generator
# Implementing below `generator` to generate an image using `z`. The function returns the generated 28 x 28 x `out_channel_dim` images.

# In[92]:

def generator(z, out_channel_dim, is_train=True):
   
    # TODO: Implement Function
    with tf.variable_scope('generator', reuse=not is_train):
        # first fully connected layer
        x1 = tf.layers.dense(z, 2*2*512)
        # reshape to start convolutional stack
        x1 = tf.reshape(x1, (-1, 2, 2, 512))
        x1 = tf.layers.batch_normalization(x1, training=is_train)
        x1 = tf.maximum(0.2 * x1, x1)
        #x1 = tf.nn.dropout(x1, keep_prob=0.8)
        #print('1st layer shape:', x1.shape)
        # 2x2x512 now
        
        x2 = tf.layers.conv2d_transpose(x1, filters=256, kernel_size=2, strides=2, padding='same')
        x2 = tf.layers.batch_normalization(x2, training=is_train)
        x2 = tf.maximum(x2 * 0.2, x2)
        #x2 = tf.nn.dropout(x2, keep_prob=0.8)
        #print('2nd layer shape:', x2.shape)
        # 4x4x256 now
        
        x3 = tf.layers.conv2d_transpose(x2, filters=128, kernel_size=4, strides=1, padding='valid')
        x3 = tf.layers.batch_normalization(x3, training=is_train)
        x3 = tf.maximum(x3 * 0.2, x3)
        #x3 = tf.nn.dropout(x3, keep_prob=0.8)
        #print('3rd layer shape:', x3.shape)
        # 7x7x128
        
        x4 = tf.layers.conv2d_transpose(x3, filters=64, kernel_size=5, strides=2, padding='same')
        x4 = tf.layers.batch_normalization(x4, training=is_train)
        x4 = tf.maximum(x4 * 0.2, x4)
        #x4 = tf.nn.dropout(x4, keep_prob=0.8)
        #print('4th layer shape:', x4.shape)
        # 14x14x64
        
        logits = tf.layers.conv2d_transpose(x4, filters=out_channel_dim, kernel_size=5, strides=2, padding='same')
        #print('logits shape:', logits.shape)
        # 28x28xout_channel_dim
        
        out = tf.tanh(logits)
        
        return out


tests.test_generator(generator, tf)


# ### Loss
# Implementing `model_loss` to build the GANs for training and calculate the loss using the following functions already implemented:
# - `discriminator(images, reuse=False)`
# - `generator(z, out_channel_dim, is_train=True)`

# In[93]:

def model_loss(input_real, input_z, out_channel_dim):
   
    # TODO: Implement Function
    g_model = generator(input_z, out_channel_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)
    #d_loss_real = tf.reduce_mean(
        #tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)*0.9))
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)*0.9))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))
    
    d_loss = d_loss_real + d_loss_fake
    
    return d_loss, g_loss
    

tests.test_model_loss(model_loss)


# ### Optimization
# Implementing `model_opt` to create the optimization operations for the GANs. I've used [`tf.trainable_variables`](https://www.tensorflow.org/api_docs/python/tf/trainable_variables) to get all the trainable variables.  

# In[94]:

def model_opt(d_loss, g_loss, learning_rate, beta1):
    #Filter the variables with names that are in the discriminator and generator scope names. 
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimization
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt
    

tests.test_model_opt(model_opt, tf)


# ## Neural Network Training
# ### Show Output
# This function shows the current output of the generator during training. It will help to determine how well the GANs is training.

# In[95]:

import numpy as np

def show_generator_output(sess, n_images, input_z, out_channel_dim, image_mode):
    cmap = None if image_mode == 'RGB' else 'gray'
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_channel_dim, False),
        feed_dict={input_z: example_z})

    images_grid = helper.images_square_grid(samples, image_mode)
    pyplot.imshow(images_grid, cmap=cmap)
    pyplot.show()


# ### Training
# Implementing `train` to build and train the GANs by using the following functions implemented:
# - `model_inputs(image_width, image_height, image_channels, z_dim)`
# - `model_loss(input_real, input_z, out_channel_dim)`
# - `model_opt(d_loss, g_loss, learning_rate, beta1)`
# 
# I've used the `show_generator_output` to show `generator` output while training. Since running `show_generator_output` for every batch drastically increased training time and increased the size of the notebook, I printed the `generator` output every 100 batches only.

# In[96]:

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
 
    # TODO: Build Model
    input_real, input_z, ln_rate = model_inputs(data_shape[1], data_shape[2], len(data_image_mode), z_dim)
    #print("train->input_real.get_shape: ", input_real.get_shape())
    #print("train->input_z.get_shape: ", input_z.get_shape())
    #print("train->learning_rate.get_shape: ", ln_rate.get_shape())
    
    
    input_real, input_z, ln_rate = model_inputs(data_shape[1], data_shape[2], data_shape[3], z_dim)
    
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[3])
    d_opt,  g_opt  = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    saver = tf.train.Saver()
    samples, losses = [], []
    steps = 0
    
    # Sample from random noise for G
    # batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch_i in range(epoch_count):
            for batch_images in get_batches(batch_size):
                # TODO: Train Model
                steps += 1
                
                # Sample from random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                # Rescale the tanh logits to between -1,1
                batch_images = batch_images * 2.0
                
                # optimizers
                _ = sess.run(d_opt, feed_dict = {input_real: batch_images, input_z: batch_z,
                                                 ln_rate:learning_rate})
                _ = sess.run(g_opt, feed_dict = {input_z: batch_z, input_real: batch_images,
                                                 ln_rate:learning_rate})
                _ = sess.run(g_opt, feed_dict = {input_z: batch_z, input_real: batch_images,
                                                 ln_rate:learning_rate})
                
                if steps %20 == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images, ln_rate:learning_rate})
                    
                    #train_loss_g = g_loss.eval({input_z: batch_z, input_real: batch_images, ln_rate:learning_rate})
                    #train_loss_g = g_loss.eval({input_z: batch_z, ln_rate:learning_rate})
                    train_loss_g = g_loss.eval({input_z: batch_z, ln_rate:learning_rate})
                    
                    
                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))
                    
                if steps %200 == 0:
                    show_generator_output(sess, 16, input_z, data_shape[3], data_image_mode)
                                                                                      
        #saver.save(sess, './checkpoints/generator.ckpt')
            


# ### MNIST
# Testing the GANs architecture on MNIST.  

# In[97]:

batch_size = 64
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5

epochs = 2

mnist_dataset = helper.Dataset('mnist', glob(os.path.join(data_dir, 'mnist/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, mnist_dataset.get_batches,
          mnist_dataset.shape, mnist_dataset.image_mode)


# ### CelebA
# Since The Generator loss is close to zero so running the GANs on CelebA.  

# In[98]:

batch_size = 64
z_dim = 100
learning_rate = 0.0002
beta1 = 0.5

epochs = 1

celeba_dataset = helper.Dataset('celeba', glob(os.path.join(data_dir, 'img_align_celeba/*.jpg')))
with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, celeba_dataset.get_batches,
          celeba_dataset.shape, celeba_dataset.image_mode)

