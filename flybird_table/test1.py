import numpy as np
import random
import os
import pygame
import time
import matplotlib.pyplot as plt
import tensorflow as tf
#print语句学习
# a=1.3
# print('hello reinforcement learning')
# print('My name is %s'%('bao zi xian er'))
# print("I'm %d years old"%(31))
# print("I'm %f meters in height"%(1.75))
# print("I'm %.2f meters in height"%1.75)
# print("老师how萌傻！")
# #if语句学习
# score =600
# if score >700:
#     print("上清华或北大！")
# else:
#     print("复读")
# score = 600
# if score >700:
#     print("上清华")
# elif score>=650:
#     print("上其他双一流大学")
# elif score > 600 or score==600:
#     print("上一本")
# else:
#     print("复读")
# #循环语句
# sum = 0
#
#
# a=[1,3,5,7,9]
# for i in a:
#     if i==1:
#         print("10以内的奇数为\n%d"%i)
#     else:
#         print(i)
# b=["天","地","玄","黄"]
# for i in b:
#     print(i)
# for i in range(100):
#     print(i)
#while语句学习
# i=0
# while i<100:
#     print(i)
#     i+=1
# i=0
# while i<100:
#     if i<50:
#         i+=1
#         continue
#     print(i)
#     i+=1
#     if i>80:
#         break
# def step(s, a):
#     s_next = s+a*0.01
#     return s_next
class maze:
    def __init__(self, dt):
        #成员变量用self
        self.dt = dt
    #成员函数
    def step(self,s,a):
        s_next = s+a*self.dt
        return s_next
if __name__=="__main__":
    maze1=maze(dt=0.01)
    s_next=maze1.step(2,3)
    print(s_next)






# x=np.linspace(1,102,102)
# y=np.linspace(1,102,102)
# s=np.array([0.5,0.8])
# ss = s[np.newaxis,:]
# print(s)
# print(ss)
# plt.figure()
# plt.plot(x,y)
# plt.show()