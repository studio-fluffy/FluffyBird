import gym
import pygame
import numpy as np
import copy
import torch
import torch.nn as nn
import math

from tensorboardX import SummaryWriter
from gym import error, spaces, utils
from gym.utils import seeding
from pygame.locals import *
import random as random

def standardAction(decission):
	return [0.0, decission[0]]

class Pipe:
    '''
    Röhre.
    ''' 
    def __init__(self, pos,  height, gap, sh):
        '''
        Setzen von Höhe height, Lücke gap, Position pos und Bildschirmhöhe sh
        '''
        self.height = height
        self.gap = gap
        self.pos = pos
        self.sh = sh

        #graphische Elemente
        self.rectU = pygame.Rect(pos,0,60,sh-height-gap)
        self.rectD = pygame.Rect(pos,sh - height,60,height)
    def render(self, context, offset):
        '''Röhre zeichnen
        https://www.pygame.org/docs/ref/draw.html#pygame.draw.rect
        '''
        pygame.draw.rect(context, pygame.Color(15,104,47), self.rectD.move(-offset +20, 0))
        pygame.draw.rect(context, pygame.Color(15,104,47), self.rectU.move(-offset +20, 0))

class Bird:
    def __init__(self, sh):
        self.Y = 250
        self.X = 80
        self.rect = pygame.Rect(self.X, sh - self.Y, 30,  30)
        self.speedY= 0
        self.speedX = 20
        self.forceX = 0.0 
        self.forceY = 0.0
        self.birdImg = pygame.image.load('sprites/sparrow/sparrow.png').convert_alpha()
        self.birdImgFlap = pygame.image.load('sprites/sparrow/sparrow_flap.png').convert_alpha()
        self.sh = sh
        self.action = standardAction
        self.ticks = 0
        self.flap = 0

    def update(self, dt, decission):
        self.action(decission, self)
        fX = self.forceX	
        fY = -10 + self.forceY

        dY = self.speedY*dt + 0.5*fY*dt*dt
        dX = self.speedX*dt + 0.5*fX*dt*dt

        self.Y += dY
        self.X += dX
        
        self.speedY += fY*dt
        self.speedX += fX*dt
	
        self.speedX = max(0.0, self.speedX)
        self.speedX = min(110.0, self.speedX)	
        self.rect = pygame.Rect(self.X, self.sh - self.Y, 30,  25)
       # self.rect.move_ip(dX, -dY)

        
        
    def reset(self):
        self.X =  80
        self.Y = 250
        self.speedY = 0
        self.speedX = 20
        self.rect = pygame.Rect(self.X, self.sh - self.Y, 30,  25)


    def render(self, context):
        threshold = math.exp(-math.pow(self.forceY,2))
        # birdImg = pygame.image.load('sprites/kit-birds.gif').convert_alpha()
        if(self.forceY <2):
#            pygame.draw.rect(context, pygame.Color(255, 0, 0, 100), self.rect.move(-self.X + 20, 0))
            context.blit(self.birdImg, (20, -self.Y + pygame.display.get_surface().get_size()[1]))
            self.ticks = 0
        else:
            self.ticks += 1
            if self.ticks > 8:
#                pygame.draw.rect(context, pygame.Color(255, 0, 0, 100), self.rect.move(-self.X + 20, 0))
                context.blit(self.birdImg, (20, -self.Y + pygame.display.get_surface().get_size()[1]))
                if self.ticks > 16:
                    self.ticks = 0
            # do rotation here
            else:
#                pygame.draw.rect(context, pygame.Color(255, 0, 255, 100), self.rect.move(-self.X + 20, 0))
                context.blit(self.birdImgFlap, (20, -self.Y + pygame.display.get_surface().get_size()[1]))


class birdEnv(gym.Env):

    metadata = {'render.modes': ['human','rgb_array']}

    screen_width = 290
    screen_height = 500

    max_FPS = 100

    gravity = 10.0
    flap_force = 20
    max_speed = 5.0
    t = 0.1
    score = 0
    birdx0 = 0
    birdy0 = screen_height * 0.5


    groundy = screen_height * 0.85

    pipes = []

    done = False

    # actions and observation space
    action_space = spaces.Discrete(2)  
    observation_space = spaces.Discrete(5)

    def __init__(self):
        pygame.init()
        self.score = 0
        self.num_pop = 0
        self.font = pygame.font.SysFont("comicsansms", 20)
        self.window = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption('Learning bird')
        #pygame.time.wait(2)
        self.fps_timer = pygame.time.Clock()
        pygame.mouse.set_visible(1)

        self.seed()
        self.img = pygame.image.load('sprites/background-day.png').convert()


 #       self.pushPipe()
 #       self.pushPipe()
        self.prand = [[250,350], [100,300],[120,130]]

        self.bird = Bird(self.screen_height)



        self.pipes = []
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()

    def playWithNet(self, net, generateFeatures, MAX_REWARD, computeReward, num_pop):
        state = self.reset()
        reward = 0.0
        done = False
        self.num_pop = num_pop
        while not done and reward < MAX_REWARD:	
            obs = torch.FloatTensor([generateFeatures(state)])
            action = self.action_space.sample()
            act_prob = net(obs).data.numpy()[0]
            acts = 0
#        if (act_prob[0] < act_prob[1]):
#            acts = 1
        #        print(acts)
            state_old = state
            state, _, done, _ = self.step(act_prob)
            reward += computeReward(state_old, state)
            self.score = reward
            self.render()
        return reward

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def setPipeIntervals(self, prand):
        self.prand = prand
    
    def setAction(self, action):
        self.bird.action = action

    def step(self, action):

        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
        pygame.event.pump()
        reward = 1

        if not self.done:
            self.bird.update(self.t, action)

            if (self.bird.Y <= 0) or (self.bird.Y >= self.screen_height):
                self.done = True



#            birdrect = pygame.Rect(self.bird.X-20 ,self.bird.Y+20,40,40)
            for pipe in self.pipes:
                #print(pipe.rectD, (self.bird.X, self.bird.Y))

                #if(pipe.rectD.collidepoint(self.bird.X, -self.bird.Y + self.screen_height) or pipe.rectU.collidepoint(self.bird.X, -self.bird.Y + self.screen_height)):
                if(self.bird.rect.colliderect(pipe.rectD) or  self.bird.rect.colliderect(pipe.rectU)):
                     self.done = True
                     reward = 0
                     #print(pipe.rectD,self.bird.X, -self.bird.Y + pygame.display.get_surface().get_size()[1] )
                     #print("collision")

                if(pipe.pos < self.bird.X -100):
                    self.pushPipe()
                    self.pipes.pop(0)

        state = self.getState()
        
        return state, 0, self.done, {},

    def getState(self):
        pipes_sorted = sorted(self.pipes, key=lambda p: p.pos)

        return {'bird': self.bird, 'pipes': pipes_sorted}



    def reset(self):
        self.done = False

        self.bird.reset()

        self.pipes = []
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()
        self.pushPipe()

        return self.getState()

    def render(self, mode="human", **kwargs):
        self.fps_timer.tick(self.max_FPS)
        self.window.fill((0, 0, 30))
        #self.window.fill(self.background)
        self.window.blit(self.img, (0,0))
        #self.window.blit(self.img_base, (0, self.groundy))

        self.bird.render(self.window)
        for pipe in self.pipes:
            pipe.render(self.window, self.bird.X )
        self.text_pop = self.font.render("Population: " + str(self.num_pop), True, (255, 255, 255))
        self.text_score = self.font.render("Score: " + str(self.score), True, (255, 255, 255))

        self.window.blit(self.text_pop,
                    (self.screen_width - self.text_pop.get_width() - 20,   self.text_pop.get_height() +10 ))

        self.window.blit(self.text_score,
                    (self.screen_width - 110,   self.text_pop.get_height() +self.text_score.get_height()  +20 ))

        pygame.display.flip()

    def spawnBird(self, type):
            self.bird = Bird()

    def pushPipe(self):
        offset = 0
        if len(self.pipes) > 0:
            lp = self.pipes[len(self.pipes) - 1]
            offset = lp.pos

        pipe = Pipe(offset + random.uniform(self.prand[0][0], self.prand[0][1]), random.uniform(self.prand[1][0], self.prand[1][1]) ,random.uniform(self.prand[2][0], self.prand[2][1]), self.screen_height)
        self.pipes.append(pipe)



