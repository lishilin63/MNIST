# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Tensorflow For Deep Learning Neural Network

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data",one_hot = True)
sample = mnist.train.images[100].reshape(28,28)
plt.imshow(sample,cmap = 'Greys')

# Set Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

n_classes = 10
n_samples = mnist.train.num_examples

n_input = 784
n_hidden_1 = 256
n_hidden_2 = 256

# Multi-Layer Perceptron Function
def multilayer_perceptron(x,weights,biases):
    '''
    x: Placeholder for data input
    weights: Dict of weights
    biases: Dict of bias values
    '''
    # First Hidden Layer with RELU Activation
    # X*W+B
    layer_1 = tf.add(tf.matmul(x,weights['h1']),biases['b1'])
    # Func(X*W+B) = RELU -> f(x) = max(0,x)
    layer_1 = tf.nn.relu(layer_1)
    
    # Second Hidden Layer
    layer_2 = tf.add(tf.matmul(layer_1,weights['h2']),biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    
    # Last Output Layer
    out_layer = tf.matmul(layer_2,weights['out'])+biases['out']
    
    return out_layer


weights = {
        'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
        'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_hidden_2,n_classes]))
        }

biases = {
        'b1':tf.Variable(tf.random_normal([n_hidden_1])),
        'b2':tf.Variable(tf.random_normal([n_hidden_2])),
        'out':tf.Variable(tf.random_normal([n_classes]))
        }
    
x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_classes])
pred = multilayer_perceptron(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

t = mnist.train.next_batch(1)
Xsamp,ysamp = t
plt.imshow(Xsamp.reshape(28,28),cmap = 'Greys')

# Run the Session
sess = tf.InteractiveSession()
init = tf.global_variables_initializer() 
sess.run(init)


# 15 loops
for epoch in range(training_epochs):
    
    # Cost
    avg_cost = 0.0
    
    total_batch = int(n_samples / batch_size) # 55000/100 = 550 
    
    for i in range(total_batch):
        
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        
        _,c = sess.run([optimizer,cost],feed_dict={x:batch_x,y:batch_y})
        
        avg_cost += c/total_batch
        
        print("Epoch: {} cost{:4f}".format(epoch+1,avg_cost))
        
print("Model has completed {} Epochs of training".format(training_epochs))



# Model Evaluation
correct_preditctions = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
print(correct_preditctions[0])  # Tensorflow will not print
correct_preditctions = tf.cast(correct_preditctions,'float')
print(correct_preditctions[0])  # Tensorflow will not print
accuracy = tf.reduce_mean(correct_preditctions)
#mnist.test.labels
#mnist.test.images
accuracy.eval({x:mnist.test.images, y:mnist.test.labels})






    
    
    
    
    
    
