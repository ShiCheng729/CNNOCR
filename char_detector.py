
import tensorflow as tf
import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time
import cv2
import numpy 
import common
import gen
import model


def predict(image):
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 36]))
    b = tf.Variable(tf.zeros([36]))
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
       
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 36])
    b_fc2 = bias_variable([36])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    init_op = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
     
    with tf.Session() as sess:
        sess.run(init_op)

        new_saver = tf.train.import_meta_graph('model2.ckpt.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint('./'))
       
        prediction=tf.argmax(y_conv,1)

        result = prediction.eval(feed_dict={x: [image],keep_prob: 1}, session=sess)

        result2 = sess.run(y_conv, feed_dict={x: [image],keep_prob: 1})

        result2 = (result2[0])
    tf.reset_default_graph()     
    return common.CHARS[result[0]],result2

def image_preproccessing(image):
    img = image.astype(numpy.float32)/255.
    img = cv2.resize(img, (28, 28))
    img = img.flatten()
    return img






