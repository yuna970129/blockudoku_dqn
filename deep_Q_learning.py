'''
network build, init
env 가져오기 e-greedy, action, reward 받기 -> 학습 안시키고 buffer에 저장, 이거 반복
위 loop 몇번 돌면 buffer에서 random.sample 해서 학습
'''

import tensorflow as tf
import numpy as np
from Blockdoku_game import *
import random
from collections import deque
import matplotlib.pyplot as plt

env=Board()

input_size = 84
output_size = len(env.action_space)

dis = 0.9
REPLAY_MEMORY = 50000
    
class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()

    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(
                tf.float32, [None,   self.input_size], name="input_x")
            
            # first layer of weights
            W1 = tf.get_variable("W1", shape=[self.input_size, h_size],
                             initializer=tf.contrib.layers.xavier_initializer())
            layer = tf.nn.tanh(tf.matmul(self._X, W1))
            # second layer of weights
            W2 = tf.get_variable("W2", shape=[h_size, self.output_size],
                                initializer=tf.contrib.layers.xavier_initializer())
      
            # Predicted reward value (Q prediction)
            self._Qpred = tf.matmul(layer, W2)
        
        # Label data type
        # we need to define the parts of the network needed for learning a 
        # policy
        self._Y = tf.placeholder(
            shape=[None, self.output_size], dtype=tf.float32)
 
        # loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        # learning
        self.train = tf.train.AdadeltaOptimizer(learning_rate=l_rate).minimize(self._loss)
 
    def predict(self, state):
        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict = {self._X: x})
 
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self.train],
                feed_dict={self._X: x_stack, self._Y: y_stack})

        
def simple_replay_train(DQN, train_batch):
    x_stack = np.empty(0).reshape(0, DQN.input_size)
    y_stack = np.empty(0).reshape(0, DQN.output_size)
    
    # get stored information from the buffer
    for state, action, reward, next_state, done in train_batch:
        Q = DQN.predict(state)
        
        # terminal?
        if done:
            Q[0, action] = reward
        else:
            # obtain the Q' values by feeding the new state through our network
            Q[0, action] = reward + dis*np.max(DQN.predict(next_state))
        
        y_stack = np.vstack([y_stack, Q])
        x_stack = np.vstack([x_stack, state])
        
    # train our network using target and predicted Q values on each episode
    return DQN.update(x_stack, y_stack)