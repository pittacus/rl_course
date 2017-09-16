# coding=utf8
import numpy as np
import tensorflow as tf
from cartpole_env import *
GAMMA = 0.95
MAX_EPISODE = 2000
RENDER = False
MAX_REWARD = 20000
class Actor:
    def __init__(self, sess, n_feature, n_actions, lr=0.001):
        self.sess=sess
        #创建前向神经网络
        self.s = tf.placeholder(tf.float32, [1,n_feature], "state")
        with tf.name_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units = 20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0.0,0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='acts_prob'
            )
        #创建损失函数
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0,self.a])
            self.loss =-tf.reduce_mean(log_prob * self.td_error)
        #创建训练函数，得到神经网络逆向传播
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def learn(self, s, a,td):
        s = s[np.newaxis,:]
        feed_dict = {self.s:s, self.a:a, self.td_error:td}
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict)
        return loss
    def choose_action(self, s):
        s = s[np.newaxis,:]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.random.choice(np.arange(probs.shape[1]),p=probs.ravel())
    def greedy_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s})
        return np.argmax(probs)
class Critic:
    def __init__(self, sess, n_feature, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32,[1,n_feature],'state')
        with tf.variable_scope('Critic'):
            #第一层网络
            l1 = tf.layers.dense(
                inputs = self.s,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
            #第二层
            self.v = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0.,.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='V'
            )
        self.v_next = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')
        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_next - self.v
            #构造损失函数
            self.loss = tf.square(self.td_error)
        #构造训练函数，对神经网络进行逆向传播
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def learn(self, s, r, s_next):
        s, s_next = s[np.newaxis,:],s_next[np.newaxis,:]
        v_next = self.sess.run(self.v,{self.s:s_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],{self.s: s, self.v_next: v_next, self.r: r})
        return td_error
class CartPole_AC:
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.cartpole = CartPoleEnv()
        self.sess = sess
        self.n_features = n_features
        self.n_actions = n_actions
        self.lr = lr
        self.RENDER = False
        self.actor = Actor(self.sess,self.n_features,self.n_actions,self.lr)
        self.critic = Critic(self.sess,self.n_features,self.lr)
        sess.run(tf.global_variables_initializer())

    def learning(self):
        for i_episode in range(MAX_EPISODE):
            s = self.cartpole.reset()
            t=0
            if self.RENDER:break
            #回报记忆空间
            track_r=[]
            while True:
                #选择动作
                a = self.actor.choose_action(s)
                #与环境交互，获得下一个状态和回报
                s_next, r, done = self.cartpole.step(a)
                if done: r=-20
                track_r.append(r)
                #对评价函数进行学习，并得到td偏差
                td_error =self.critic.learn(s, r, s_next)
                #对动作函数进行学习
                self.actor.learn(s,a,td_error)
                #智能体推进到下一步
                s = s_next
                t+= 1
                if done:
                    ep_rs_sum = sum(track_r)
                    if ep_rs_sum>MAX_REWARD:
                        self.RENDER=True
                    if i_episode %100 ==0:
                        print(i_episode,ep_rs_sum)
                    # print(ep_rs_sum)
                    break
    def learning_test(self):
        s = self.cartpole.reset()
        while True:
            a = self.actor.greedy_action(s)
            self.cartpole.render()
            s_next, r, done = self.cartpole.step(a)
            if done:
                break
            s = s_next

if __name__=="__main__":
    sess = tf.Session()
    cartpole1= CartPole_AC(sess,4,2,lr=0.001)
    cartpole1.learning()
    cartpole1.learning_test()
