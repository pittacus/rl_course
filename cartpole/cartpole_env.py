# coding=utf8
import pygame
import numpy as np
from load import *
from pygame.locals import *
import math
import time

class CartPoleEnv:
    def __init__(self):
        self.actions = [0,1]
        self.state =np.random.uniform(-0.05, 0.05,size=(4,) )
        self.steps_beyond_done = 0
        self.viewer = None
        #设置帧率
        self.FPSCLOCK = pygame.time.Clock()
        self.screen_size = [400, 300]
        self.cart_x=200
        self.cart_y=200
        self.theta =-1.5
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass =(self.mass_cart+self.mass_pole)
        self.length = 0.5
        self.pole_mass_length = (self.mass_pole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02
        #角度阈值
        self.theta_threshold_radians = 12*2*math.pi/360
        #x方向阈值
        self.x_threhold =2.4
    def reset(self):
        n = np.random.randint(1,1000,1)
        np.random.seed(n)
        self.state = np.random.uniform(-0.05, 0.05,size=(4,) )
        self.steps_beyond_done = 0
        # print(self.state)
        return np.array(self.state)
    def step(self,action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action ==1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        #动力学方程
        temp = (force+self.pole_mass_length * theta_dot *theta_dot* sintheta)/self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp)/(self.length * (4.0/3.0-self.mass_pole * costheta * costheta/self.total_mass))
        xacc = temp - self.pole_mass_length * thetaacc * costheta /self.total_mass
        #积分得到状态量
        x = x+self.tau * x_dot
        x_dot = x_dot +self.tau * xacc
        theta = theta +self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)
        #根据更新的状态判断是否结束
        done = x < -self.x_threhold or x > self.x_threhold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        done = bool(done)
        #设置回报
        if not done:
            reward = 1.0
            self.steps_beyond_done = self.steps_beyond_done+1
        # elif self.steps_beyond_done is None:
        #     self.steps_beyond_done = 0
        #     reward = 1.0
        else:
            # self.steps_beyond_done = self.steps_beyond_done+1
            # print(self.steps_beyond_done)
            reward = 0.0
        return np.array(self.state), reward, done
    def render(self):
        screen_width = self.screen_size[0]
        screen_height = self.screen_size[1]
        world_width = self.x_threhold * 2
        scale = screen_width/world_width
        state = self.state
        # print(state)
        self.cart_x = 200+scale * state[0]
        self.cart_y = 200
        self.theta = state[2]
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size,0,32)
            self.background = load_background()
            self.pole = load_pole()
            # print(self.pole.)
            #画背景
            self.viewer.blit(self.background,(0,0))
            self.viewer.blit(self.pole,(195,80))
            pygame.display.update()
        #循环绘图
        self.viewer.blit(self.background,(0,0))
        #画线
        pygame.draw.line(self.viewer, (0,0,0),(0,200),(400,200))
        #画圆
        # pygame.draw.circle(self.viewer,(250,0,0),(200,200),1)
        #画矩形
        pygame.draw.rect(self.viewer, (250,0,0),(self.cart_x-20,self.cart_y-15,40,30))
        # cart1=pygame.Rect(self.cart_x-20,self.cart_y-15,40,30)
        pole1=pygame.transform.rotate(self.pole, -self.theta*180/math.pi)
        if self.theta > 0:
            pole1_x = self.cart_x-5*math.cos(self.theta)
            pole1_y = self.cart_y-80*math.cos(self.theta)-5*math.sin(self.theta)
        else:
            pole1_x = self.cart_x+80*math.sin(self.theta)-5*math.cos(self.theta)
            pole1_y = self.cart_y-80*math.cos(self.theta)+5*math.sin(self.theta)

        self.viewer.blit(pole1, (pole1_x, pole1_y))
        # self.viewer.blit(self.pole1, (self.cart_x-5, self.cart_y-80))

        # pygame.transform.rotate(self.pole,130)
        # self.viewer.blit(cart1)
        # pygame.transform.rotate(cart,10)
        pygame.display.update()
        self.FPSCLOCK.tick(30)

if __name__=="__main__":
    cartpole = CartPoleEnv()
    cartpole.reset()
    cartpole.render()
    i=0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
        while True:
            random_num = np.random.random()
            if random_num > 0.5:
                action = 1
            else:
                action = -1
            s,r,t=cartpole.step(action)
            if t==1:
                print(cartpole.steps_beyond_done)
                cartpole.reset()

            # cartpole.theta +=0.1
            # cartpole.cart_x+=10
            time.sleep(0.02)
            cartpole.render()
