# coding=utf8
import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import shutil
from cartpole_env import *
N_WORKERS = multiprocessing.cpu_count()
print(N_WORKERS)
MAX_EP_STEP = 400
MAX_GLOBAL_EP = 800
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001
LR_C = 0.001
GLOBAL_RUNNING_R=[]
GLOBAL_EP = 0
N_S = 4
N_A = 1
A_BOUND=[-10,10]



class ACNet:
    def __init__(self, scope,globalAC=None):
        #全局网络
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32,[None, N_S], 'S')
                self.build_net()
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')
        #局部网络，计算损失
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32,[None, N_A],'A')
                self.v_target = tf.placeholder(tf.float32,[None, 1],'Vtarget')
                mu, sigma, self.v = self.build_net()
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('wrap_a_out'):
                    mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mu, sigma)
                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()
                    self.exp_v = ENTROPY_BETA*entropy+exp_v
                    self.a_loss = tf.reduce_mean(-self.exp.v)
                with tf.name_scope('choose_a'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1),axis=0), A_BOUND[0],A_BOUND[1])
                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope+'/critic')
                    #计算局部网络对参数的梯度
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
            with tf.name_scope('sync'):
                #将全局变量的参数应用到每个worker中
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC)]
                #将局部梯度应用到全局参数中
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
    def build_net(self):
        w_init = tf.random_normal_initializer(0.,.1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh,kernel_initializer=w_init,name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        return mu, sigma, v
    def update_global(self, feed_dict):
        SESS.run([self.update_a_op,self.update_c_op],feed_dict)
    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])
    def choose_action(self, s):
        s = s[np.newaxis,:]
        return SESS.run(self.A, {self.s:s})[0]
class Worker:
    def __init__(self,name, globalAC):
        self.cartpole = CartPoleEnv()
        self.name = name
        #创建AC网络
        self.AC = ACNet(name,globalAC)
    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [],[],[]
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s= self.cartpole.reset()
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                if self.name == 'W_0':
                    self.cartpole.render()
                #选择动作
                a = self.AC.choose_action(s)
                s_, r, done = self.cartpole.step(a)
                done = True if ep_t == MAX_EP_STEP-1 else False
                r/=10
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_ =0
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s:s_[np.newaxis,:]})[0,0]
                    buffer_v_target=[]
                    for r in buffer_r[::-1]:
                        v_s_=r+GAMMA*v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s),np.vstack(buffer_a),np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s:buffer_s,
                        self.AC.a_his:buffer_a,
                        self.AC.v_target:buffer_v_target
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s,buffer_a,buffer_r=[],[],[]
                    self.AC.pull_global()
                s = s_
                total_step+=1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9*GLOBAL_RUNNING_R[-1]+0.1*ep_r)
                    GLOBAL_EP += 1
                    break

if __name__=="__main__":
    SESS = tf.Session()
    with tf.device("\cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
        workers = []
        for i in range(N_WORKERS):
            #创建worker
            i_name = 'W_%i'% i
            workers.append(Worker(i_name, GLOBAL_AC))#创建workers
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())
    worker_threads = []
    for worker in workers:
        job = lambda:worker.work()
        #创建一个线程
        t = threading.Thread(target=job)
        #开始一个线程
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
