# coding=utf8
import numpy as np
import tensorflow as tf
from pendulum_env import *
GAMMA=0.99
MAX_EPISODE = 2000
RENDER = False
MAX_REWARD = -50
MAX_EP_STEPS = 1000

class Actor:
    def __init__(self, sess, n_feature, action_bound, lr=0.01):
        self.sess = sess
        self.n_feature = n_feature
        #创建前向神经网络
        self.s = tf.placeholder(tf.float32,[1,self.n_feature],"state")
        with tf.name_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.,.1),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'l1'
            )
            #第二层神经网络
            mu = tf.layers.dense(
                inputs=l1,
                units=1,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0.,.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='mu'
            )
            sigma = tf.layers.dense(
                inputs = l1,
                units = 1,
                activation=tf.nn.softplus,
                kernel_initializer=tf.random_normal_initializer(0.,.1),
                bias_initializer=tf.constant_initializer(1.0),
                name='sigma'
            )
        global_step = tf.Variable(0, trainable=False)
        self.mu, self.sigma = tf.squeeze(mu*2), tf.squeeze(sigma+0.1)
        #定义带参数的正态分布
        self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
        self.action = tf.clip_by_value(self.normal_dist.sample(1), action_bound[0],action_bound[1])
        #创建输入占位符
        self.a = tf.placeholder(tf.float32, None, name="act")
        self.td_error = tf.placeholder(tf.float32, None, name="td_error")
        with tf.variable_scope('exp_v'):
            log_prob = self.normal_dist.log_prob(self.a)
            self.loss = log_prob*self.td_error
            self.loss+=self.normal_dist.entropy()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.loss, global_step)
    def learn(self, s, a, td):
        s = s[np.newaxis,:]
        feed_dict = {self.s: s, self.a: a, self.td_error:td}
        _, loss = self.sess.run([self.train_op, self.loss],feed_dict)
        return loss
    def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.action, {self.s:s})
class Critic:
    def __init__(self, sess, n_feature,lr=0.01):
        self.sess = sess
        with tf.name_scope('inputs'):
            self.s = tf.placeholder(tf.float32,[1,n_feature],"state")
            self.v_next = tf.placeholder(tf.float32,[1,1],name="v_next")
            self.r = tf.placeholder(tf.float32, name='r')
        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=30,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0.,.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='l1'
            )
        self.v = tf.layers.dense(
            inputs=l1,
            units=1,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(0.,.1),
            bias_initializer=tf.constant_initializer(0.1),
            name='V'
        )
        with tf.variable_scope('squared_TD_error'):
            self.td_error = tf.reduce_mean(self.r + GAMMA*self.v_next -self.v)
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)
    def learn(self, s, r, s_next):
        s, s_next = s[np.newaxis, :], s_next[np.newaxis, :]
        v_next = self.sess.run(self.v,{self.s:s_next})
        td_error, _ = self.sess.run([self.td_error, self.train_op],{self.s:s, self.v_next:v_next, self.r:r})
        return td_error
class Pendulum_AC:
    def __init__(self, sess, n_features, lr=0.01):
        self.pendulum = PendulumEnv()
        self.sess =sess
        self.n_features = n_features
        self.lr=lr
        self.RENDER = False
        A_bound =np.array([-2,2])
        self.actor = Actor(self.sess, self.n_features,A_bound,lr=self.lr)
        self.critic = Critic(self.sess, n_feature=self.n_features,lr=self.lr)
        self.sess.run(tf.global_variables_initializer())
    def learning(self):
        for i_episode in range(MAX_EPISODE):
            s = self.pendulum.reset()
            t=0
            if self.RENDER:
                print(i_episode)
                break
            ep_rs=[]
            while True:
                #选择动作
                a = self.actor.choose_action(s)
                s_next, r, done = self.pendulum.step(a)
                # print(s_next)
                r/=10
                td_error = self.critic.learn(s,r,s_next)
                self.actor.learn(s,a,td_error)
                s=s_next
                t+=1
                ep_rs.append(r)
                if t>MAX_EP_STEPS:
                    ep_rs_sum = sum(ep_rs)
                    print(ep_rs_sum)
                    if ep_rs_sum>MAX_REWARD:
                        print(ep_rs_sum)
                        self.RENDER=True
                    # if i_episode%50==0
                    #     print()
                    break
    def learning_test(self):
        s = self.pendulum.reset()
        t=0
        while True:
            self.pendulum.render()
            a = self.actor.choose_action(s)
            s_next, r, done = self.pendulum.step(a)
            if t>MAX_EP_STEPS:
                break
            s = s_next
            t+=1
            print(a)



if __name__=="__main__":
    sess = tf.Session()
    pendulum1 = Pendulum_AC(sess,3,lr=0.01)
    pendulum1.learning()
    pendulum1.learning_test()
