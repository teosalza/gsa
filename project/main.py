from itertools import filterfalse
from random import randint
import numpy as np
import math
import random
import time


class Agent:
    def __init__(self,id,pos,fit = 0,vel = 0, mass = 1, acc = 0):
        self.id = id
        self.pos = pos
        self.vel = vel
        self.pos1 = 0
        self.vel1 = 0
        self.fit = fit
        self.mass = mass
        self.acc = acc

    def __str__(self):
        return "[Id: {0}, Pos:{1} , Vel: {2}, Fit: {3}, Mass: {4}, Acc: {5}]".format(self.id, self.pos, self.vel, self.fit, self.mass, self.acc)


def getFitnessF1(x):
    fit = np.sum(np.array(x)**2)
    return fit
def getGCostant(iteraion, total_iteration):
    alpha = 20
    G0 = 100
    return G0*np.exp(-alpha*(iteraion / total_iteration))
def getMass(agent_fitness, bestFitness, worstFitness):
    if agent_fitness == worstFitness: 
        agent_fitness += 0.0001
    mi = (agent_fitness - worstFitness) / (bestFitness - worstFitness)
    if(mi == 0.0): print(str(agent_fitness) + "--"+str(worstFitness))
    # return (mi / totalMass)
    return mi
def getSingleMass(agent_fitness, bestFitness, worstFitness):
    return  (agent_fitness - worstFitness) / (bestFitness - worstFitness)
def getAcceleration(agent,population_array,gCostant, dimension):
    filtered_population = [el for el in population_array if el.id != agent.id]
    acc = np.zeros(dimension)  
    Fi = 0
    
    for d in range(dimension):
        for (idx,j) in enumerate(filtered_population):
            R = math.sqrt((agent.pos[d] - j.pos[d])**2)
            Fij = gCostant * ((agent.mass * j.mass) / (R + np.finfo(float).eps)) * (j.pos[d] - agent.pos[d])
            Fi += random.random() * Fij
        # acc[d] = (Fi / agent.mass)
        acc[d] = (Fi )
    return acc
def getVelocityV1(agent,dimension):
    v1=[None] * dimension
    for d in range(dimension):
        if agent.vel == 0:
            v1[d] = (agent.acc[d])
        else:
            v1[d] = (random.random() * agent.vel[d] + agent.acc[d])
    return v1
def getPositionPos1(agent,dimension):
    pos1 = [None] * dimension
    for d in range(dimension):
        pos1[d] = (agent.pos[d] + agent.vel1[d])
    pos1 =np.clip(pos1,-100,100)
    return pos1

#variable definition
population_array = []
high_bound = 100
low_bond = -100
dimension = 30
population_number = 50
number_iterations = 1000
agent_fitness = 0
bestFitness = 0
worstFitness = 0
bestAgent = ""
worstAgent = ""
#generate population
pos_generation = np.random.uniform(0,1,(population_number,dimension))*(high_bound-low_bond)+low_bond

for (idx,pos) in enumerate(pos_generation):
    tmp_mass = Agent(idx+1,pos)
    population_array.append(tmp_mass)

#Cicle for number of iteration
for i in range(number_iterations):
    #Evaluation Fitness each agent, bestFitness, worstFitness
    for agent in  population_array:
        if(i > 0 ):
            #update velocity set velocity at t+1 as velocity at time t
            agent.vel = agent.vel1
            agent.vel1 = 0
            agent.pos = agent.pos1
            agent.pos1 = 0 
        #Calculate fitness score
        agent_fitness = getFitnessF1(agent.pos)
        agent.fit = agent_fitness
    
    bestFitness = np.min([e.fit for e in population_array])
    worstFitness = np.max([e.fit for e in population_array])
    # print("Best "+str(bestFitness))
    # print("Worst: "+str(worstFitness))
    #calculate G costant function
    G = getGCostant(i,number_iterations)

    #Calculate M for each agent
    for agent in population_array:
        #calculate Mass
        Mi = getMass(agent.fit, bestFitness, worstFitness)
        agent.mass = Mi

    totalMass = sum([el.mass for el in population_array])
    for agent in population_array:
        agent.mass = agent.mass / totalMass

    #Calculate a, pos(i+1) and vel(i+1) for each agent
    for agent in  population_array:
        #calculate Acceleration
        ai = getAcceleration(agent,population_array,G, dimension)
        agent.acc = ai
        #calculate velocity at time t+1
        vel1 = getVelocityV1(agent,dimension)
        agent.vel1 = vel1
        
        #calculate position at time t+1
        pos1 = getPositionPos1(agent,dimension)
        agent.pos1 = pos1
    print(['Iteration nr: '+ str(i+1)+ ' best fitness score: ' + str(bestFitness)])
    # print('Best agent: '+str(bestAgent))  
    #print([e.pos for e in population_array])
print("finito")     
   
