# coding=utf8
import pygame
import numpy as np
from load import *
from pygame.locals import *
import math
import time

class PendulumEnv:
    def __init__(self):
        self.max_spped = 8
        self.max_torque = 2
        self.dt = 0.05
        self.viewer = None
        #设置帧率
        self.FPSCLOCK = pygame.time.Clock()
        self.screen_size = [400,300]
        self.x = 200
        self.y = 200
        self.rotate_angle = 0
        self.theta =0
        self.gravity = 9.8
        self.state=np.array([0,0])
        self.normal_angle = 0
    def step(self,u):
        theta, thetadot = self.state
        # print(theta)
        self.theta = self.state[0]
        m = 1
        l=1
        dt = self.dt
        g=self.gravity
        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        costs = self.angle_normalize(theta)**2+0.1*thetadot**2+0.001*u**2
        new_thetadot = thetadot-(3*g/(2*l)*np.sin(theta+np.pi)+3./(m*l**2)*u)*dt
        # print(new_thetadot)
        new_theta = theta + new_thetadot*dt
        # print(theta)
        new_thetadot = np.clip(new_thetadot, -self.max_spped, self.max_spped)
        self.state = np.array([new_theta, new_thetadot])
        # print(new_theta)
        # print(self.state)
        # print(self.get_obs())
        return self.get_obs(),-costs, False
    def reset(self):
        n = np.random.randint(1, 1000, 1)
        np.random.seed(0)
        high = np.array([np.pi, 1])
        self.state = np.random.uniform(low=-high, high=high)
        return self.get_obs()
    def get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta),np.sin(theta),thetadot])
    def angle_normal(self, theta):
        normal_angle = 0
        if theta<0:
            normal_angle = theta
            while normal_angle<0:
                normal_angle += 2 * math.pi
        if theta>2*math.pi:
            normal_angle = theta
            while normal_angle>2*math.pi:
                normal_angle-=2*math.pi
        if theta<=2*math.pi and theta>=0:
            normal_angle = theta
        return normal_angle
    def angle_normalize(self,theta):
        normalized_angle =self.angle_normal(theta)-np.pi
        return normalized_angle
    def render(self):
        screen_width = self.screen_size[0]
        if self.viewer is None:
            pygame.init()
            self.viewer = pygame.display.set_mode(self.screen_size,0,32)
            self.background = load_background()
            self.pole = load_pole()
            #画背景
            self.viewer.blit(self.background,(0,0))
            self.viewer.blit(self.pole,(195,80))
            pygame.display.update()
        self.viewer.blit(self.background,(0,0))
        pygame.draw.circle(self.viewer, (0, 0, 0), (200, 200), 5)
        #将当前角度转化为旋转角度
        self.rotate_angle =self.angle_normal(self.theta)
        trans_angle = self.rotate_angle*180/math.pi
        pole_x = 0
        pole_y = 0
        if trans_angle>=0 and trans_angle<90:
            pole_x = self.x-5*math.cos(self.rotate_angle)
            pole_y = self.y-5*math.sin(self.rotate_angle)
        if trans_angle>=90 and trans_angle<180:
            pole_x = self.x-5*math.sin(self.rotate_angle-math.pi/2)
            pole_y = self.y-80*math.sin(self.rotate_angle-math.pi/2)-5*math.cos(self.rotate_angle-math.pi/2)
        if trans_angle>=180 and trans_angle<270:
            pole_x = self.x-80*math.sin(self.rotate_angle-math.pi)-5*math.cos(self.rotate_angle-math.pi)
            pole_y = self.y -80*math.cos(self.rotate_angle-math.pi)-5*math.sin(self.rotate_angle-math.pi)
        if trans_angle>=270 and trans_angle<=360:
            pole_x = self.x-80*math.cos(self.rotate_angle-3*math.pi/2)-5*math.sin(self.rotate_angle-3*math.pi/2)
            pole_y = self.y-5*math.cos(self.rotate_angle-3*math.pi/2)

        pole1=pygame.transform.rotate(self.pole, trans_angle)
        self.viewer.blit(pole1,(pole_x, pole_y))
        pygame.display.update()
        self.FPSCLOCK.tick(30)

if __name__=="__main__":
    pendulum = PendulumEnv()
    pendulum.render()
    pendulum.reset()
    # cartpole.render()
    i=0
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                exit()
        # while True:
            # pendulum.theta += 0.04
            # i+=1
            # if i>1000:
            #     break
            # time.sleep(0.05)
            # pendulum.reset()

            # pendulum.render()
            # pendulum.step(-2)
            # print(pendulum.state)
            # pendulum.render()
