
from __future__ import print_function
import numpy as np
import tensorflow as tf
import random
import os,sys,pickle
from PIL import Image

# Parameters
learning_rate = 0.001
training_iters = 61991
batch_size = 100
display_step = 10
n_input = 10000 #  (img shape: 100*50)
n_classes = 2
dropout = 0.75 # Dropout, probability to keep units

#Import
file_data1 = np.load('train_data.npz')
file_target1 = np.load('train_target.npz')
train_data = file_data1['arr_0']
train_target = file_target1['arr_0']
train_data.astype(float)
train_target.astype(float)
train_tmp = 1-train_target
train_target = np.concatenate((train_target,train_tmp),axis=0)
train_target = np.reshape(train_target, (n_classes, len(train_data)))
train_target = train_target.transpose()
train_data = np.concatenate((train_data,train_data[1:batch_size,]),axis=0)
train_target = np.concatenate((train_target,train_target[1:batch_size,]),axis=0)

file_data2 = np.load('test_data.npz')
file_target2 = np.load('test_target.npz')
test_data = file_data2['arr_0']
test_target = file_target2['arr_0']
test_data.astype(float)
test_target.astype(float)
test_tmp = 1-test_target
test_target = np.concatenate((test_target,test_tmp),axis=0)
test_target = np.reshape(test_target, (n_classes, len(test_data)))
test_target = test_target.transpose()
test_data = np.concatenate((test_data,test_data[1:batch_size,]),axis=0)
test_target = np.concatenate((test_target,test_target[1:batch_size,]),axis=0)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 100, 100, 1])
    
    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    
    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)
    
    # Convolution Layer
    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=5)
    
    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)
    
    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 16])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 16, 32])),
    'wc3': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([5*5*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([16])),
    'bc2': tf.Variable(tf.random_normal([32])),
    'bc3': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Keep training until reach max iterations
    yy = 0
    while yy < 10:
        print ("Iteration run " + str(yy))
        step = 1
        while step * batch_size < training_iters:
            #print ("Process: " + "{:.3f}".format(batch_size*step/float(training_iters)))
            batch_x = train_data[((step-1)*batch_size):(step*batch_size)]
            batch_y = train_target[((step-1)*batch_size):(step*batch_size)]
            #batch_y = np.reshape(batch_y, (batch_size, n_classes))
            #print ("Target: " + str(batch_y))
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
            if step % display_step == 0:
            #if step > 0:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,y: batch_y,keep_prob: 1.})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                if acc > 0.78 and yy > 0:
                    step = training_iters
                    yy = 10
            step += 1
        yy += 1
    print("Optimization Finished!")
    

    test_x = test_data
    test_y = test_target
    #test_y = np.reshape(test_y,(batch_size, n_classes))
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: test_x,
                   y: test_y,
                   keep_prob: 1.}))

































