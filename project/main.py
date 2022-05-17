from random import randint
import numpy as np
import math
import random


class Agent:
    def __init__(self,id,pos,fit = 0,vel = 0, mass = 0, acc = 0):
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
    fit = np.sum(x**2)
    return fit
def getGCostant(iteraion, total_iteration):
    alpha = 20
    G0 = 100
    return G0*np.exp(-alpha*(iteraion / total_iteration))
def getMass(agent_fitness, bestFitness, worstFitness, population_array, agent):
    mi = (agent_fitness - worstFitness) / (bestFitness - worstFitness)
    sum_array = sum([el.fit for el in population_array])
    # print("["+str(mi)+","+str(sum_array)+"]")
    return (mi / sum_array)
def getAcceleration(agent,population_array,gCostant):
    filtered_population = [el for el in population_array if el.id != agent.id]
    Fi = 0
    for j in filtered_population:
        R = math.sqrt((agent.pos - j.pos)**2)
        Fij = gCostant * ((agent.mass * j.mass) / (R + np.finfo(float).eps)) * (agent.pos - j.pos)
        Fi += random.random() * Fij
    acc = Fi / agent.mass
    return acc
def getVelocityV1(agent):
    v1 = random.random() * agent.vel + agent.acc  
    return v1
def getPositionPos1(agent):
    pos1 = agent.pos + agent.vel1
    return pos1

#variable definition
population_array = []
high_bound = 100
low_bond = -100
dimension = 1
population_number = 50
number_iterations = 1000
bestFitness = 0
worstFitness = 0
bestAgent = ""
worstAgent = ""
#generate population
pos_generation = np.random.uniform(0,1,population_number)*100

for (idx,pos) in enumerate(pos_generation):
    tmp_mass = Agent(idx+1,pos)
    population_array.append(tmp_mass)


for i in range(number_iterations):

    #Evaluation Fitness each agent, bestFitness, worstFitness
    for agent in  population_array:

        if(i > 0 ):
            #update velocity set velocity at t+1 as velocity at time t
            agent.vel = agent.vel1
            agent.vel1 = 0
            agent.pos = agent.pos1
            agent.pos = 0 
        

        #Calculate fitness score
        agent_fitness = getFitnessF1(agent.pos)
        agent.fit = agent_fitness
        print(agent.fit)

        #Get best and worst fitness value
        if agent_fitness > bestFitness : 
            bestFitness = agent_fitness
            bestAgent = agent
        if agent_fitness < worstFitness : 
            worstFitness = agent_fitness
            worstAgent = agent
        
    #calculate G costant function
    G = getGCostant(i,number_iterations)

    #Calculate M and a, pos(i+1) and vel(i+1) for each agent
    for agent in  population_array:
        #calculate Mass
        Mi = getMass(agent.fit, bestFitness, worstFitness, population_array, agent)
        agent.mass = Mi
        #calculate Acceleration
        ai = getAcceleration(agent,population_array,G)
        agent.acc = ai
        #calculate velocity at time t+1
        vel1 = getVelocityV1(agent)
        agent.vel1 = vel1
        #calculate position at time t+1
        pos1 = getPositionPos1(agent)
        agent.pos1 = pos1

    # print(['Iteration nr: '+ str(i+1)+ ' best fitness score: ' + str(bestFitness)]);
    # print('Best agent: '+str(bestAgent))  
    #print([e.pos for e in population_array])
print("finito")     
   
