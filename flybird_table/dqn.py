# coding=utf8
import numpy as np
import random
import pygame
import time
import matplotlib.pyplot as plt
import tensorflow as tf
from yuanyang_env import YuanYangEnv

class Dqn:
    def __init__(
            self,
            yuanyang,
            n_features,
            learning_rate=0.01,
            reward_decay=0.90,
            epsilon=0.2,
            replace_target_iter=300,
            memory_size=100,
            batch_size = 32,
            epsilon_greedy_increment=None,
            output_graph=False
    ):
        self.learning_rate = learning_rate
        self.epsilon_max = epsilon
        self.yuanyang = yuanyang
        self.actions = yuanyang.actions
        self.gamma = reward_decay
        #动作空间的维数
        self.n_actions = len(yuanyang.actions)
        #输入空间的维数为2
        self.n_features = n_features
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_greedy_increment
        # self.epsilon = 0 if epsilon_greedy_increment is not None else self.epsilon_max
        self.epsilon = 0.2
        self.learn_step_counter = 0
        #记录损失函数
        self.cost=[]
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self.batch_memory=[]
        self.build_net()
        #声明一个会话
        self.sess = tf.Session()
        #初始化会话
        self.sess.run(tf.global_variables_initializer())


    def build_net(self):
        # 创建前向神经网络
        def build_forward_layers(s, c_names, n_l1, w_initializer, b_initializer):
            #创建第一层
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1',[self.n_features, n_l1], initializer=w_initializer,collections=c_names)
                b1 = tf.get_variable('b1',[1,n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s,w1)+b1)
            #创建第二层
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions],initializer=w_initializer,collections=c_names)
                b2 = tf.get_variable('b2',[1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1,w2)+b2
            return out
        ###########利用前向神经网络分别构建值函数的计算式，损失函数，训练过程
        # 定义前向神经网络的输入
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        #根据输入构建动作值函数的计算式Q(s),输入为状态s，输出为(Q(s,a1)，Q(s,a2），Q(s,a3),Q(s,a4))
        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer=['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 100, \
                                                       tf.random_normal_initializer(0.,0.02),tf.constant_initializer(0.01)
            #创建输出
            self.q_eval = build_forward_layers(self.s, c_names, n_l1, w_initializer, b_initializer)
        #创建目标网络，目标网络为：target=r+argmax(Q(s_next,a；theta_)),所以计算目标网络需要输入r, s_
        #先得到后继状态的输出值，计算公式为Q(s_; theta_),这里theta_存到target_net_parames
        self.s_next = tf.placeholder(tf.float32,[None, self.n_features], name='s_next')
        with tf.variable_scope('target_net'):
            c_names = ['target_net_parames', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_forward_layers(self.s_next, c_names,n_l1,w_initializer,b_initializer)
        #计算q_target, 计算公式为：r+argmax(Q(s_next,a；theta_)),所以额外的输入为立即回报r
        self.r = tf.placeholder(tf.float32,[None, ], name='r')
        with tf.variable_scope('q_target'):
            if self.r == -100 or self.r==100:
                self.q_target = r
            else:
                q_target = self.r + self.gamma * tf.reduce_max(self.q_next,axis=1, name='Qmax_s_next')
                self.q_target = tf.stop_gradient(q_target)
        #计算当前动作的值函数，因为前面的q_eval是一个4*1的向量，所以要选出当前动作所对应的那个值函数。
        self.a = tf.placeholder(tf.int32, [None, ], name='a')
        #动作a所对应的值函数计算公式为：sum(Q(s)*one_hot(a))
        with tf.variable_scope('q_eval'):
            a_one_hot = tf.one_hot(self.a, depth=self.n_actions, dtype=tf.float32)
            self.q_eval_a = tf.reduce_sum(self.q_eval*a_one_hot, axis=1)
        #有了当前值函数，目标值函数，就可以定义损失函数了，计算公式为：sum(quare(q_target-q_eval_a))/N
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval_a,name='TD_error'))
        #有了损失函数，就可以进行逆向传播了
        with tf.variable_scope('train'):
            self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
    #定义数据储存函数
    def store_transition(self, s, a, r, s_next):
        #如果没有memory_counter则创建memory_counter并初始化为0
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_next))
        #用新的数据替换最老的数据
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter+=1
    #定义epsilon_greedy策略
    def epsilon_greedy_policy(self, observation):
        #将观测数组结构转化为矩阵，因为神经网络的输入是矩阵
        observation = observation[np.newaxis,:]
        #
        if np.random.uniform()<1-self.epsilon:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s:observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    #定义贪婪策略
    def greedy_policy(self, observation):
        # 将观测数组结构转化为矩阵，因为神经网络的输入是矩阵
        observation = observation[np.newaxis, :]
        action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(action_value)
        return action
    def replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t,e) for t, e in zip(t_params, e_params)])
    def sample_datas(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        self.batch_memory = self.memory[sample_index, :]
        # return batch_memory
    def state_trans_to_state(self,s):
        i = int(s / 10)
        j = s % 10
        new_s = np.array([40*j,30*i])
        return new_s
    def dqn_learning(self):
        step =0
        for episode in range(1000):
            s = self.yuanyang.reset()
            #action为0,1,2,3;a为 'e','s','w','n'
            action = self.epsilon_greedy_policy(self.state_trans_to_state(s))
            a = self.actions[action]
            t=False
            count=0
            while False==t and count<50:
            # while count < 50:
                s_next, r, t = self.yuanyang.transform(s, a)
                #将环境的状态转换为dqn所用的数据结构
                # s_next = self.state_trans_to_state(s_next)
                #将环境返回值存储到记忆池中
                self.store_transition(self.state_trans_to_state(s), action, r, self.state_trans_to_state(s_next))
                #每采集五个点更新一次权值
                if (step > 200) and (step % 5 == 0):
                    #首先检查下目标网络是否需要替换
                    if self.learn_step_counter % self.replace_target_iter == 0:
                        self.replace_target_params()
                    #从记忆池中采样数据以便用于训练
                    self.sample_datas()
                    #调用神经网络反向传播，训练一次数据
                    _, cost = self.sess.run([self.train_op, self.loss], feed_dict={self.s:self.batch_memory[:, :self.n_features],
                                                                                   self.a:self.batch_memory[:,self.n_features],
                                                                                   self.r:self.batch_memory[:,self.n_features+1],
                                                                                   self.s_next:self.batch_memory[:,-self.n_features:]})
                    self.cost.append(cost)
                    #学习次数增加一次
                    self.learn_step_counter+=1
                #智能体转到下一步
                s = s_next
                # s = self.state_trans_to_state(s)
                action = self.epsilon_greedy_policy(self.state_trans_to_state(s))
                a = self.actions[action]
                step+=1
                # print(step)
                count+=1
                # print(count)
                # if self.learn_step_counter%100==0:
                #     yuanyang.bird_male_position = yuanyang.state_to_position(s)
                #     yuanyang.render()


if __name__=="__main__":
    yuanyang = YuanYangEnv()
    agent = Dqn(yuanyang, n_features=2)
    #智能体进行学习
    agent.dqn_learning()
    print(len(agent.cost))
    plt.figure()
    plt.plot(np.linspace(1, len(agent.cost), len(agent.cost)), agent.cost)
    plt.show()
    for i in range(100):
        action = agent.greedy_policy(agent.state_trans_to_state(i))
        a = yuanyang.actions[action]
        print('%d->%s\t' % (i, a))
        print(agent.sess.run(agent.q_eval,feed_dict={agent.s:agent.state_trans_to_state(i)[np.newaxis,:]}))
    print(agent.memory)
    for i in range(len(agent.cost)):
        print(agent.cost[i])
    #测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        action = agent.greedy_policy(agent.state_trans_to_state(s))
        a = yuanyang.actions[action]
        print('%d->%s\t' % (s, a))
        print(agent.sess.run(agent.q_eval, feed_dict={agent.s: agent.state_trans_to_state(s)[np.newaxis,:]}))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
