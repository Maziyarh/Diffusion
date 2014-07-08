'''
Created on Oct 14, 2013

@author: Maziyar
'''
import numpy as np
import scipy as sp
from scipy.misc import comb
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.mlab as mlab
from matplotlib import rc
rc('text', usetex=True)

N = 100; # number of nodes in the graph
p = 0.05; # probability of forming an edge between two randomly selected nodes (for Erdos-Renyi)
watts_avg_connectivity = 3; # average connectivity (for Watts-Strogatz)
p_rewiring = 0.1; # prob. of rewiring for Watts-Strogatz
m = 1; # number of edges to attach from a new node to existing nodes
is_graph_connected = False
while (not is_graph_connected):
    G = nx.erdos_renyi_graph(N, p)
    #G = nx.watts_strogatz_graph(N, watts_avg_connectivity, p_rewiring)
    #G = nx.barabasi_albert_graph(N, m)
    is_graph_connected = nx.is_connected(G)
#plt.figure(1)
#nx.draw_random(G)

# Compute degree sequence of the graph
A = nx.adjacency_matrix(G)
node_degree = np.sum(A.transpose(), 1) # row-sum
node_degree = np.array([int(node_degree[i]) for i in xrange(0, len(node_degree))]); # just convert to a 1-D array
max_degree = np.unique(node_degree)[-1]
degree_sequence = np.array([float(sum(node_degree==k)) for k in xrange(1,max_degree+1) ]) # from 1 to max_degree
P = degree_sequence/float(N) # degree distribution
print("The sum of degree distribution = %d"%sum(P))
k_avg = sp.r_[1:max_degree+1].dot(P) # average connectivity <k>
print("The average connectivity = %d"%k_avg)
#plt.figure(2)
#plt.plot(xrange(1, max_degree+1), P)
#plt.show()

# Each node, has an associated state
s = sp.zeros(N) # N = no. of nodes
num_active_seeds = int(0.1*N) # number of active seeds
id_active_seeds = np.random.permutation(N) # randomly permute the ids of the nodes to choose which ones get active
id_active_seeds = id_active_seeds[0:num_active_seeds] # set num_active_seeds in the population to be active
s[id_active_seeds] = 1 

active_node_degree = node_degree[id_active_seeds]
active_degree_sequence = np.array([sum(active_node_degree==k) for k in xrange(1,max_degree+1) ]) # from 1 to max_degree
degree_sequence_modified = np.copy(degree_sequence)
degree_sequence_modified[mlab.find(degree_sequence==0)] = 1
rho = active_degree_sequence/degree_sequence_modified
#plt.figure(2)
#plt.plot(xrange(1, max_degree+1), rho)
#plt.show()

C = 1; # the highest possible cost of adoption of a technology
b = 1; # mutual benefit of a pairwise adoption
nu = 0.2;
delta = 0.1;

T = 500; # time period of simulation
rho_network_t = np.zeros(T)
rho_network_t[0] = rho.dot(P)

rho_network_k_t = np.zeros([max_degree, T]) 
rho_network_k_t[:,0] = rho

theta_network_t = np.zeros(T)
theta_network_t[0] = np.sum(sp.r_[1:max_degree+1]*P*rho)/k_avg

for t in xrange(1,T): # from time t=1 as we already have initial state at t=0
   
    s_next = np.copy(s); 
    active_nodes = mlab.find(s==1)
    num_active_nodes = len(active_nodes)
    # First, some delta rate of the active nodes become susceptible
    num_active_to_susceptible = int(delta*num_active_nodes)
    active_nodes = np.random.permutation(active_nodes) # randomly permute
    active_to_susceptible_nodes = active_nodes[0:num_active_to_susceptible]
    s_next[active_to_susceptible_nodes] = 0 # return to susceptible state
    
    #Next, some nu rate of the susceptible nodes become active
    susceptible_nodes = mlab.find(s==0)
    num_susceptible_nodes = len(susceptible_nodes)
    num_susceptible_to_active = int(nu*num_susceptible_nodes)
    susceptible_nodes = np.random.permutation(susceptible_nodes)
    susceptible_to_active_nodes = susceptible_nodes[0:num_susceptible_to_active]
    # Of these nodes who are considering adoption of technology, a network based choice is now made
    for node in susceptible_to_active_nodes:
        ki = node_degree[node]; # susceptible node degree
        neighbor_nodes = mlab.find(A[node,:]==1) # these are the neighbors of node i
        ai = np.sum(s[neighbor_nodes]) # number of neighbors who are active
        ci= sp.rand()*C; # uniform cost of adoption ~U[0,C]
        if (b*ai > ci):
            s_next[node]=1
    
    # Now, compute the active degree distribution
    active_nodes = mlab.find(s_next==1)
    for node in active_nodes:
        k = node_degree[node]
        rho_network_k_t[k-1, t] += 1 # as we start from degree = 1
    rho_network_k_t[:,t] = rho_network_k_t[:,t]/degree_sequence_modified
    rho_network_t[t] = rho_network_k_t[:,t].dot(P)
    theta_network_t[t] = np.sum(sp.r_[1:max_degree+1]*P*rho_network_k_t[:,t])/k_avg
    s=np.copy(s_next)
    if (np.mod(t,100)==0):
        print("Agent-based simulation time step %d of %d"%(t,T))
print("Completed agent-based network simulation\n")

dT = 1e-0
ode_T = sp.r_[0:T:dT]
rho_ode_t = sp.zeros(len(ode_T))
rho_ode_k_t = sp.zeros([max_degree, len(ode_T)])
theta_ode_t = sp.zeros(len(ode_T))

rho_ode_t[0] = rho.dot(P)
rho_ode_k_t[:,0] = rho
theta_ode_t[0] = np.sum(sp.r_[1:max_degree+1]*P*rho)/k_avg
g_k_a = sp.zeros([max_degree, max_degree+1]) # for every k, and every 0 <= a <= k
g = sp.zeros(max_degree)
f = lambda a: a/float(C) if (a<=C) else 1
# Now, we simulate the differential equation
for t in xrange(1,len(ode_T)):
    # We compute g(k) for every k 
    for k in xrange(1,max_degree+1):
        for a in xrange(0, max_degree+1):
            g_k_a[k-1,a] = nu*f(a)*comb(k,a)*(theta_ode_t[t-1]**a)*((1-theta_ode_t[t-1])**(k-a))
        g[k-1] = sum(g_k_a[k-1,:])
        delta_rho_k_t = (-rho_ode_k_t[k-1,t-1]*delta + (1-rho_ode_k_t[k-1,t-1])*g[k-1])*dT
        rho_ode_k_t[k-1,t] = rho_ode_k_t[k-1,t-1] + delta_rho_k_t
    rho_ode_t[t] = rho_ode_k_t[:,t].dot(P)
    theta_ode_t[t] = np.sum(sp.r_[1:max_degree+1]*P*rho_ode_k_t[:,t])/k_avg
    if (np.mod(t,dT*100)==0):
        print("Mean-field time step %d of %d"%(t,len(ode_T)))
print("Completed mean-field simulation\n")

plt.figure(1)
plt.plot(xrange(0,T), theta_network_t, 'b', label="Agent-based")
plt.plot(ode_T, theta_ode_t, 'r', label="ODE")
plt.legend(loc=4, mode="expand")
plt.title(r'\theta(t)')

plt.figure(2)
plt.plot(xrange(0,T), rho_network_t, 'b', label="Agent-based")
plt.plot(ode_T, rho_ode_t, 'r', label="ODE")
plt.legend(loc=4, mode="expand")
plt.title(r'\rho(t)')

plt.figure(3)

for i in xrange(1,10):
    plotnum = 330+i
    plt.subplot(plotnum)
    plt.plot(xrange(0,T), rho_network_k_t[i-1,:], 'b', label="Agent-based")
    plt.plot(ode_T, rho_ode_k_t[i-1,:], 'r', label="ODE")

plt.show()