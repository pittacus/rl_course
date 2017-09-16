# coding=utf8
import numpy as np
import tensorflow as tf
from cartpole_env import *


class CartPole_PG:
    def __init__(self, cartpole, learning_rate=0.02, reward_decay=0.95):
        #动作空间
        self.actions = cartpole.actions
        self.n_actions = len(self.actions)
        self.RENDER = False
        self.DISPLAY_REWARD_THRESHOLD=20000
        #状态空间
        self.state = cartpole.state
        self.n_state = len(self.state)
        #学习速率
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        #轨迹的观测值，动作值和回报
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        #创建策略网络
        self.build_net()
        #启动一个默认的会话
        self.sess = tf.Session()
        #初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())
    #创建前向神经网络
    def build_net(self):
        with tf.name_scope('input'):
            #创建占位符作为输入
            self.obs = tf.placeholder(tf.float32,[None, self.n_state], name="observations")
        #创建第一层
        layer = tf.layers.dense(
            inputs=self.obs,
            units = 10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1'
        )
        #创建第二层
        all_act = tf.layers.dense(
            inputs=layer,
            units=self.n_actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )
        #最后一层，softmax层
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')
        #定义损失函数为L=plogq*V,其中有两个新的输入：输入动作和值函数，所以先定义占位符
        self.sample_actions = tf.placeholder(tf.int32,[None, ], name="actions_num")
        self.vt = tf.placeholder(tf.float32, [None, ], name = "action_value")
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.sample_actions)
            loss = tf.reduce_mean(neg_log_prob*self.vt)
        #定义训练，神经网络后向传播
        with tf.name_scope('train'):
            self.train_op =tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        #定义如何选择行为：即利用当前策略网络对行为进行采样
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:observation[np.newaxis,:]})
        #按照当前的概率分布进行采样
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
    #定义贪婪策略
    def greedy(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.obs:observation[np.newaxis,:]})
        action = np.argmax(prob_weights.ravel())
        return action
    #定义存储，将一个回合的状态，动作和回报都放在一起，s1,a1,r1,s2,a2,r2,s3,a3,s3......
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    #处理回报，对回报进行累计，并归一化
    def discount_and_norm_reward(self):
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        #对回报进行累加
        for t in reversed(range(0,len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        #归一化处理
        discounted_ep_rs-=np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
    def learning(self):
        for i_episode in range(500):
            #重启环境,得到重置的状态
            observation = cartpole.reset()
            if self.RENDER:break
            while True:
                if self.RENDER: cartpole.render()
                #产生一个动作，与环境进行交互
                action = self.choose_action(observation)
                # print(action)
                # cartpole.render()
                #调用step函数时，cartpole的状态在step函数中也得到了更新
                observation_next, reward, done = cartpole.step(action)
                #将下一步观测，动作和回报储存起来
                self.store_transition(observation, action, reward)
                #如果本回试验结束，则计算累积回报，并进行学习
                if done:
                    #看看当前所有的累积回报是多少
                    ep_rs_sum = sum(self.ep_rs)
                    if ep_rs_sum >self.DISPLAY_REWARD_THRESHOLD:self.RENDER = True
                    # if
                    #计算折扣累积回报
                    discounted_ep_rs_norm = self.discount_and_norm_reward()
                    #进行一次学习
                    self.sess.run(self.train_op, feed_dict={
                        self.obs: np.vstack(self.ep_obs),
                        self.sample_actions:np.array(self.ep_as),
                        self.vt:discounted_ep_rs_norm
                    })
                    print("episode:", i_episode,"rewards:", int(ep_rs_sum))
                    # print(discounted_ep_rs_norm)
                    # print(self.ep_rs)
                    #清空episode数据，为下次存储数据做准备
                    self.ep_obs,self.ep_as,self.ep_rs=[],[],[]
                    # print("the current total return is %d"%(ep_rs_sum))
                    break
                #智能体探索一步
                observation = observation_next
    def learning_test(self):
        observation = cartpole.reset()
        while True:
            # 产生一个动作，与环境进行交互
            action = self.greedy(observation)
            # 调用step函数时，cartpole的状态在step函数中也得到了更新
            observation_next, reward, done = cartpole.step(action)
            if done:
                break
            observation = observation_next
            #print(action)
            cartpole.render()

if __name__=="__main__":
    cartpole = CartPoleEnv()
    agent = CartPole_PG(cartpole)
    reward = agent.learning()
    print("I have learnt")
    print(reward)
    agent.learning_test()
