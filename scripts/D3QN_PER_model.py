# -*- coding: utf-8 -*-
"""
D3QN PER Model

@author: ziqi han
"""

import tensorflow.compat.v1 as tf
# import tensorlayer as tl 
tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
REPLAY_SIZE = 1000
GAMMA = 0.9  # Discount Factor
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.05
EPISODES = 5000


class DQN():
    def __init__(self, observation_width, observation_height, action_space, model_file, log_file):
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.beta_init = 0.4
        self.beta = self.beta_init
        self.TD_list = np.array([])
        self.learning_rate = 0.001
        self.eps = 0.00001
        self.alpha = 0.6

        # deque is a list that first-in & first-out
        self.replay_buffer = deque()
        self.create_Q_network()
        self.create_updating_method()
        self.epsilon = INITIAL_EPSILON
        self.model_path = model_file + "/save_model.ckpt"
        self.model_file = model_file
        self.log_file = log_file

        config = tf.ConfigProto()                   
        config.gpu_options.per_process_gpu_memory_fraction = 0.6    
        self.session = tf.InteractiveSession(config=config) 
        # Init session
        if os.path.exists(self.model_file):
            print("model exists , load model\n")
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self.model_path)
        else:
            print("model don't exists , create new one\n")
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        self.writer = tf.summary.FileWriter(self.log_file, self.session.graph)
        # tensorboard --logdir=logs_gpu --host=127.0.0.1
        self.merged = tf.summary.merge_all()
        
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(tf.constant(0.01, shape=shape))
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    def create_Q_network(self):
        # the function to create the network
        # there are two networks, action_value and target_action_value
        # these two networks has same architecture
        
        with tf.name_scope('inputs'):
            self.state_input = tf.placeholder("float", [None, self.state_h, self.state_w, 1])

        with tf.variable_scope('current_net'):
            W_conv1 = self.weight_variable([5,5,1,32])
            b_conv1 = self.bias_variable([32])
            W_conv2 = self.weight_variable([5,5,32,64])
            b_conv2 = self.bias_variable([64])

            w_fc1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])         
            b_fc1 = self.bias_variable([512])
            
            w_fc2 = self.weight_variable([512, 256])                      
            b_fc2 = self.bias_variable([256])                                                                         
            
            w_fc3_1 = self.weight_variable([256, 1])                      
            b_fc3_1 = self.bias_variable([1])                                                                    
            w_fc3_2 = self.weight_variable([256, self.action_dim] )                    
            b_fc3_2 = self.bias_variable([self.action_dim])                        

            h_conv1 = tf.nn.relu(self.conv2d(self.state_input, W_conv1) + b_conv1)
            # self.state_w * self.state_h * 32
            h_pool1 = self.max_pool_2x2(h_conv1)
            # self.state_w/2 * self.state_h/2 * 32
            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
            # self.state_w/2 * self.state_h/2 * 64
            h_pool2 = self.max_pool_2x2(h_conv2)
            # self.state_w/4 * self.state_h/4 * 64
 
            # 传递给全连接层的卷积层的输出必须在全连接层接受输入之前进行flatten
            h_conv2_flat = tf.reshape(h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])

            h_layer_1  = tf.nn.relu(tf.matmul(h_conv2_flat, w_fc1) + b_fc1) 
            h_layer_1  = tf.nn.dropout(h_layer_1, 0.5)
            h_layer_2  = tf.nn.relu(tf.matmul(h_layer_1, w_fc2) + b_fc2) 
            h_layer_2  = tf.nn.dropout(h_layer_2, 0.5)
            h_layer_3_state  = tf.matmul(h_layer_2, w_fc3_1) + b_fc3_1
            h_layer_3_action = tf.matmul(h_layer_2, w_fc3_2) + b_fc3_2  
            h_layer_3_advantage = tf.subtract(h_layer_3_action, tf.reduce_mean(h_layer_3_action)) 
                                                     
            Q_value = tf.add(h_layer_3_state, h_layer_3_advantage)                  
            self.Q_value = tf.nn.dropout(Q_value, 0.5)

        with tf.variable_scope('target_net'):
            t_W_conv1 = self.weight_variable([5,5,1,32])
            t_b_conv1 = self.bias_variable([32])
            t_W_conv2 = self.weight_variable([5,5,32,64])
            t_b_conv2 = self.bias_variable([64])

            t_w_fc1 = self.weight_variable([int((self.state_w/4) * (self.state_h/4) * 64), 512])         
            t_b_fc1 = self.bias_variable([512])
            
            t_w_fc2 = self.weight_variable([512, 256])                      
            t_b_fc2 = self.bias_variable([256])                                                                         
            
            t_w_fc3_1 = self.weight_variable([256, 1])                      
            t_b_fc3_1 = self.bias_variable([1])                                                                    
            t_w_fc3_2 = self.weight_variable([256, self.action_dim] )                    
            t_b_fc3_2 = self.bias_variable([self.action_dim]) 

            t_h_conv1 = tf.nn.relu(self.conv2d(self.state_input, t_W_conv1) + t_b_conv1)   
            t_h_pool1 = self.max_pool_2x2(t_h_conv1)    
            t_h_conv2 = tf.nn.relu(self.conv2d(t_h_pool1, t_W_conv2) + t_b_conv2)
            t_h_pool2 = self.max_pool_2x2(t_h_conv2) 

            t_h_conv2_flat = tf.reshape(t_h_pool2, [-1,int((self.state_w/4) * (self.state_h/4) * 64)])

            t_h_layer_1  = tf.nn.relu(tf.matmul(t_h_conv2_flat, t_w_fc1) + t_b_fc1) 
            t_h_layer_1  = tf.nn.dropout(t_h_layer_1, 0.5)
            t_h_layer_2  = tf.nn.relu(tf.matmul(t_h_layer_1, t_w_fc2) + t_b_fc2) 
            t_h_layer_2  = tf.nn.dropout(t_h_layer_2, 0.5)
            t_h_layer_3_state  = tf.matmul(t_h_layer_2, t_w_fc3_1) + t_b_fc3_1
            t_h_layer_3_action = tf.matmul(t_h_layer_2, t_w_fc3_2) + t_b_fc3_2  
            t_h_layer_3_advantage = tf.subtract(t_h_layer_3_action, tf.reduce_mean(t_h_layer_3_action)) 
                                                     
            target_Q_value = tf.add(t_h_layer_3_state, t_h_layer_3_advantage) 
            self.target_Q_value = tf.nn.dropout(target_Q_value, 0.5)

        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # Replace target_net's parameters with current_net's parameters
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def create_updating_method(self):
        # Update current_net's parameters
        # def loss_and_train(self):
        
        # one-hot vector
        self.action_input = tf.placeholder(tf.float32, shape = [None, self.action_dim])
        # TD aim value
        self.y_input = tf.placeholder(tf.float32, shape = [None])

        # Q_value of action
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # ################################################## PER ############################################################
        self.w_is = tf.placeholder(tf.float32, shape = [None])
        self.TD_error_tf = tf.subtract(Q_action, self.y_input)

        loss = tf.reduce_sum(tf.multiply(self.w_is, tf.square(Q_action - self.y_input)))
        ###################################################################################################################

        # Loss-Plot
        tf.summary.scalar('loss',loss)
        with tf.name_scope('train_loss'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(loss)


    def Choose_Action(self, state):
        # def select_action(self, progress, sess, observation_stack, state_stack, Epsilon):

        Q_value = self.Q_value.eval(feed_dict={ self.state_input: [state] })[0]
        
        # Epsilon Greedy
        if random.random() <= self.epsilon:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    def Store_Data(self, state, action, reward, next_state, done):
        # Store data in Replay Buffer
        # def Experience_Replay(...):
        # def prioritized_minibatch(self):

        one_hot_action = np.zeros([self.action_dim])
        one_hot_action[action] = 1

        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
            self.TD_list = np.delete(self.TD_list, 0)
        #################################################### PER ############################################################
        Q_batch = self.target_Q_value.eval(feed_dict = { self.state_input: [next_state] })
        if done:
            y = [reward]
        else:
            y = [reward + GAMMA * np.max(Q_batch)]

        TD_error = self.TD_error_tf.eval(feed_dict = {self.y_input: y,self.action_input: [one_hot_action],self.state_input: [state]})[0]
        print(pow((abs(TD_error) + self.eps), self.alpha))
        self.TD_list = np.append(self.TD_list, pow((abs(TD_error) + self.eps), self.alpha))
        # print(self.TD_list)
        #################################################### PER ############################################################

    def Train_Network(self, BATCH_SIZE, num_step):
        # def train(self, minibatch, w_batch, batch_index):

        # Update TD_error list                                           

        TD_normalized = self.TD_list / np.linalg.norm(self.TD_list, 1) 
        TD_sum = np.cumsum(TD_normalized) 
                                                                                                                    
        # Get importance sampling weights                                                                              
        weight_is = np.power((REPLAY_SIZE * TD_normalized), - self.beta)                                    
        weight_is = weight_is / np.max(weight_is)                                                                      
                                                                                                                    
        # Select mini batch and importance sampling weights                                                            
        minibatch = []                                                                                                 
        batch_index = []                                                                                               
        w_batch = []                                                                                                   
        for i in range(BATCH_SIZE): 
            while(1):                                                                               
                rand_batch = random.random()
                TD_index = np.nonzero(TD_sum >= rand_batch)[0]
                print(TD_sum,TD_normalized)
                if TD_index != [] :
                    TD_index = TD_index[0]
                    break
             
            batch_index.append(TD_index)                                                                               
            w_batch.append(weight_is[TD_index])                                                                        
            # minibatch.append(self.Replay_memory[TD_index])                                                           
            minibatch.append(np.array(self.replay_buffer)[TD_index])

        state_batch      = [data[0] for data in minibatch]
        action_batch     = [data[1] for data in minibatch]
        reward_batch     = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        y_batch = []
        # Double DQN
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # Calculate TD aim value
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                # Final state
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={ 
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch,
            self.w_is: w_batch})

        TD_error_batch = self.session.run(self.TD_error_tf,feed_dict={
        self.y_input: y_batch,
        self.action_input: action_batch,
        self.state_input: state_batch,
        self.w_is: w_batch})

        # Update TD_list
        for i_batch in range(len(batch_index)):
            self.TD_list[batch_index[i_batch]] = pow((abs(TD_error_batch[i_batch]) + self.eps), self.alpha)

        # Update Beta
        self.beta = self.beta + (1 - self.beta_init) / EPISODES

        if num_step % 100 == 0:
            # Loss-Plot
            result = self.session.run(self.merged,feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch,
            self.w_is: w_batch})
            self.writer.add_summary(result, num_step)

    def Update_Target_Network(self):
        # Update Target Q Network
        # def assign_network_to_target(self):

        self.session.run(self.target_replace_op)
    
    def save_model(self):
        self.save_path = self.saver.save(self.session, self.model_path)
        print("Save to path:", self.save_path)
    
    def action(self, state):
        # use for test
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])