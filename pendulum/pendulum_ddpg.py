# coding=utf8
import numpy as np
import tensorflow as tf
from pendulum_env import *
MEMORY_CAPACITY = 7000
BATCH_SIZE = 32
REPLACE_ITER_A = 500
REPLACE_ITER_C = 300
GAMMA = 0.9
LR_A = 0.01
LR_C = 0.01
MAX_EPISODES = 700
MAX_EP_STEPS = 400
RENDER = False
class Pendulum_DDPG:
    def __init__(self, pendulum,a_dim, s_dim, a_bound,):
        self.pendulum = pendulum
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim*2+a_dim+1),dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter=0,0
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.S = tf.placeholder(tf.float32,[None, s_dim],'s')
        self.S_=tf.placeholder(tf.float32, [None, s_dim],'s_')
        self.R = tf.placeholder(tf.float32,[None,1],'r')
        self.RENDER=False
        with tf.variable_scope('Actor'):
            self.a = self.build_a(self.S, scope='eval', trainable=True)
            a_ = self.build_a(self.S_, scope='target',trainable=False)
        with tf.variable_scope('Critic'):
            q = self.build_c(self.S, self.a, scope='eval', trainable=True)
            q_=self.build_c(self.S_,a_,scope='target',trainable=False)
        #网络参数
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='Critic/target')
        q_target = self.R + GAMMA*q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)
        a_loss = -tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss,var_list=self.ae_params)
        self.sess.run(tf.global_variables_initializer())
    #定义确定性策略
    def choose_action(self,s):
        return self.sess.run(self.a, {self.S:s[np.newaxis,:]})[0]
    def learn(self):
        if self.a_replace_counter % REPLACE_ITER_A == 0:
            self.sess.run([tf.assign(t,e) for t,e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % REPLACE_ITER_C == 0:
            self.sess.run([tf.assign(t,e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter+=1
        self.c_replace_counter+=1
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices,:]
        bs = bt[:, :self.s_dim]
        ba = bt[:,self.s_dim:self.s_dim+self.a_dim]
        br = bt[:, -self.s_dim-1:-self.s_dim]
        bs_=bt[:,-self.s_dim:]
        self.sess.run(self.atrain,{self.S:bs})
        self.sess.run(self.ctrain,{self.S:bs, self.a:ba,self.R:br,self.S_:bs_})
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s,a,[r],s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index,:] = transition
        self.pointer += 1
    def build_a(self,s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s,30,activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a',trainable=trainable)
            return tf.multiply(a,self.a_bound, name='scaled_a')
    def build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s',[self.s_dim,n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a',[self.a_dim,n_l1],trainable=trainable)
            b1 = tf.get_variable('b1',[1,n_l1],trainable=trainable)
            # print(s)
            net = tf.nn.relu(tf.matmul(s, w1_s)+tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net,1,trainable=trainable)
    def learning(self):
        for i_episode in range(MAX_EPISODES):
            s = self.pendulum.reset()
            ep_reward = 0
            for j in range(MAX_EP_STEPS):
                var = 3
                if self.RENDER:
                    self.pendulum.render()
                a = self.choose_action(s)
                #添加随机项到动作，以便探索
                a = np.clip(np.random.normal(a,var),-2,2)
                s_, r, done = self.pendulum.step(a)
                self.store_transition(s, a, r/10, s_)
                if self.pointer > MEMORY_CAPACITY:
                    var *= 0.9995
                    self.learn()
                s = s_
                ep_reward+=r
                if j==MAX_EP_STEPS-1:
                    print('Episode:',i_episode,'Reward: %i'%int(ep_reward), 'Explore:%.2f' %var)
                    if ep_reward > -10: self.RENDER = True
                    break
if __name__=="__main__":
    a_bound = 2
    pendulum1 = PendulumEnv()
    agent = Pendulum_DDPG(pendulum1,a_dim=1,s_dim=3,a_bound=a_bound)
    agent.learning()
