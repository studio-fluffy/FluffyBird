import copy
import numpy as np

import torch
import torch.nn as nn

def mutate_net(net, NOISE_STD):
    new_net = copy.deepcopy(net)
    for p in new_net.parameters():
        noise_t = torch.tensor(np.random.normal(size = p.data.size()).astype(np.float32))
        p.data += NOISE_STD * noise_t
    return new_net

class Population ():
    def __init__(self, POPULATION_SIZE, obs_size, action_size, computeReward, net):
        self.POPULATION_SIZE = POPULATION_SIZE
        self.obs_size = obs_size
        self.action_size = action_size
        self.net = net
        self.nets = [copy.deepcopy(net) for _ in range(POPULATION_SIZE)]
        self.computeReward = computeReward

    def evaluate_on_env (self, env, generateFeatures, MAX_REWARD):
        self.population = []
        for net in self.nets:
            state = env.reset()
            reward = 0.0
            done = False
            while not done and reward <MAX_REWARD:
                obs = torch.FloatTensor([generateFeatures(state)])
                #action = env.action_space.sample()

                act_prob = net(obs).data.numpy()[0]
                acts = 0
 #               if(act_prob[0] < act_prob[1]):
 #                   acts = 1
                state_old = state
                state, done, _ = env.step(act_prob)
                #env.render()
                reward += self.computeReward(state_old, state)

                #env.render()
            #print(reward)

            self.population.append([reward, net])

        self.population.sort(key=lambda p: p[0], reverse=True)
  #      print("--")
  #      for p in self.population:
  #          print (p[0])
  #      print("--")
    def playWithFittest(self, env, generateFeatures, MAX_REWARD, num_pop):
#        print(self.population[0][0])
        playWithPopulation(env, self.population[0], generateFeatures, MAX_REWARD, self.computeReward, num_pop)

def playWithPopulation(env, pop,generateFeatures, MAX_REWARD, computeReward, num_pop ):
    state = env.reset()
    reward = 0.0
    done = False
    net = pop[1]
    while not done and reward < MAX_REWARD:
        obs = torch.FloatTensor([generateFeatures(state)])
        action = env.action_space.sample()
        act_prob = net(obs).data.numpy()[0]
        acts = 0
#        if (act_prob[0] < act_prob[1]):
#            acts = 1

        #        print(acts)
        state_old = state

        state, done, _ = env.step(act_prob)
        reward += computeReward(state_old, state)
        env.render(str(num_pop), str("%.1f" % reward))
    print (reward)

def mutate_population(pop, PARENTS_COUNT, NOISE_STD):
        pop_mut = Population(0, pop.obs_size, pop.action_size, pop.computeReward, pop.net)
        pop_mut.nets = [(copy.deepcopy(pop.population[0][1]))]
        for _ in range(len(pop.population) - 1):
            parent_idx = np.random.randint(0, PARENTS_COUNT)
            parent = pop.population[parent_idx][1]
            net_mut = mutate_net(parent,NOISE_STD )
            pop_mut.nets.append(net_mut)
        return pop_mut
