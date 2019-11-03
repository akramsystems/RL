'''
#################################################################################   
Author:         Ali Akram

Description:    This is a Library for Reinforcement Learning Algorithms

Parameters:     S       - State Space |S|-Vector
                P       - Transitional Probability Matrix |S|x|S|-Matrix
                R       - Reward Vector |S|-Vector
                Gamma   - Discount Factor  CONSTANT
                
#################################################################################  
'''
import numpy as np
from math import inf

''' Evalutaing Value Function Methods '''


def MonteCarloEvaluation(M, s, t, N, generateEpisode):
    '''
        Approximates the value for a state at time t
        for finate horizon cases

        M : [S, P, R, Gamma]
        s : starting State
        t : time step of starting State
        N : number of total time steps Finite Horizon
        
        generateEpisode : used to simulate the futureStates
        given time t and state s
    '''
    i = 0; Gt = 0;
    [S, P, R, Gamma] = M
    while i != N:
        
        futureStates = generateEpisode(s,t) # returns vector of size H - 1 - t
        g = 0
        for j in range(t,N-1):
            g += gamma**(j-t) * R[j]
        Gt += g
        i += 1
    Vt = Gt / N # average value 
    return Vt # update value  function

def AnalyticSolution(P,R,Gamma):
    '''
        Uses Linear Algebra to generate the value function
    '''
    P = np.asarray(P); R = np.asarray(R); I = np.identity(len(R));
    try:
        inv= np.linalg.inv(I - Gamma*P)
        V = inv*R
        return V
    except expression as identifier:
        print("There was a problem with getting Calculating the error")
        return ["There was a problem with getting Calculating the error"]
    
def iterativeValue(M,e):
    '''
        This is used for finding the Value function for MRP with an
        infinite Horizon
        e:  'episolon'
            our tolerance telling us when we converge
            to a solution
    '''
    [P, R, Gamma] = M
    P = np.asarray(P)
    R = np.asarray(R)
    n = len(R)

    Vp = [0]*n; V = [inf]*n

    while abs(V-Vp) < e:
        V = Vp
        for s in range(n):
            Vp[s] = R[s] + Gamma*P.dot(Vp)
    
    return V









