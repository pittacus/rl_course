# coding=utf8
import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
from yuanyang_env import *
from yuanyang_env import YuanYangEnv

class Td_Sarsa:
    def __init__(self, YuanYangEnv):
        #值函数的初始值
        self.qvalue=np.zeros((len(YuanYangEnv.states),len(YuanYangEnv.actions)))
    #定义贪婪策略
    def greedy_policy(self,YuanYangEnv, qfun, state):
        amax=qfun[state,:].argmax()
        return YuanYangEnv.actions[amax]
    #定义epsilon贪婪策略
    def epsilon_greedy_policy(self,YuanYangEnv, qfun, state, epsilon):
        amax = qfun[state, :].argmax()
        # 概率部分
        if np.random.uniform() < 1 - epsilon:
            # 最优动作
            return YuanYangEnv.actions[amax]
        else:
            return YuanYangEnv.actions[int(random.random() * len(YuanYangEnv.actions))]
    #找到动作所对应的序号
    def find_anum(self,YuanYangEnv,a):
        for i in range(len(YuanYangEnv.actions)):
            if a==YuanYangEnv.actions[i]:
                return i

    def sarsa(self, YuanYangEnv, num_iter, alpha, epsilon):
        for iter in range(num_iter):
            #随机初始化状态
            s = YuanYangEnv.reset()
            #随机选初始动作
            a = YuanYangEnv.actions[int(random.random()*len(YuanYangEnv.actions))]
            t = False
            count = 0
            while False==t and count < 200:
                #与环境交互得到下一个状态
                s_next, r, t = YuanYangEnv.transform(s, a)
                a_num = self.find_anum(YuanYangEnv, a)
                if t == True:
                    q_target = r
                else:
                    # 下一个状态处的最大动作
                    a1 = self.epsilon_greedy_policy(YuanYangEnv, self.qvalue, s_next, epsilon)
                    a1_num = self.find_anum(YuanYangEnv, a1)
                    # qlearning的更新公式
                    q_target = r + YuanYangEnv.gamma * self.qvalue[s_next, a1_num]
                    # 利用td方法更新动作值函数
                self.qvalue[s, a_num] = self.qvalue[s, a_num] + alpha * (q_target - self.qvalue[s, a_num])
                # YuanYangEnv2.bird_male_position = YuanYangEnv2.state_to_position(s)
                # YuanYangEnv2.render()
                # time.sleep(1)
                # 转到下一个状态
                s = s_next
                a = self.epsilon_greedy_policy(YuanYangEnv, self.qvalue, s, epsilon)
                count += 1
        return self.qvalue
if __name__=="__main__":
    yuanyang = YuanYangEnv()
    agent = Td_Sarsa(yuanyang)
    qvalue=agent.sarsa(yuanyang, num_iter=1000, alpha=0.1, epsilon=0.1)
    #打印学到的值函数
    print(qvalue)
    ##########################################
    #测试学到的策略
    flag = 1
    s = 0
    # print(policy_value.pi)
    step_num = 0
    # 将最优路径打印出来
    while flag:
        a = agent.greedy_policy(yuanyang,qvalue,s)
        print('%d->%s\t' % (s, a))
        yuanyang.bird_male_position = yuanyang.state_to_position(s)
        yuanyang.render()
        time.sleep(0.2)
        step_num += 1
        s_, r, t = yuanyang.transform(s, a)
        if t == True or step_num > 200:
            flag = 0
        s = s_
    # print('optimal policy is \t')
    # print(policy_value.pi)
    # print('optimal value function is \t')
    # print(policy_value.v)
    # xx = np.linspace(0, len(policy_value.v) - 1, 101)
    # yy = policy_value.v
    # plt.figure()
    # plt.plot(xx, yy)
    # plt.show()
    # # 将值函数的图像显示出来
    # z = []
    # for i in range(100):
    #     z.append(1000 * policy_value.v[i])
    # zz = np.array(z).reshape(10, 10)
    # plt.figure(num=2)
    # plt.imshow(zz, interpolation='none')
    # plt.show()
