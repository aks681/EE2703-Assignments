
# coding: utf-8

# # Assignment 6
# #### - Akshay Anand (EE16B046)
# 
# The code can be called in the commandline in the following format:
# 
# python Assgn6.py n=value M=value Msig=Value nk=Value u0=Value p=Value seed=Value
# 
# where all the arguments are optional and value is some number


import numpy as np    
import matplotlib.pyplot as plt
import pandas as pd    # pandas for showing in tabular form
import sys



n = 100    # Spatial grid size
M = 10     # no. of electrons injected per turn
Msig = 2   # standard deviation of the distribution electrons injected	
nk = 500   # no. of turns to simulate
u0 = 7.0   # threshold velocity
p = 0.5   # probability that ionization will occur
seedValue = 2   # A seed for random number generator


args = sys.argv[1:]  # This is because argv[0] is the program name itself which is not required
argList = {}         # Dictionary declared
for i in args:
    i = i.split('=')    # 'varname=value' string is split into varname and value
    argList[i[0]] = float(i[1])   # The dictionary is updated with values from commandline


if 'n' in argList:
    n = argList['n']
if 'M' in argList:
    M = argList['M']
if 'nk' in argList:
    nk = argList['nk']
if 'u0' in argList:
    u0 = argList['u0']
if 'p' in argList:
    p = argList['p']
if 'Msig' in argList:
    Msig = argList['Msig']
if 'seed' in argList:
    seedValue = argList['seed']


xx = np.zeros(n*M)  # electron position
u = np.zeros(n*M)   # electron velocity
dx = np.zeros(n*M)  # electron displacement
np.random.seed((int)(seedValue))   # A seed given so that uniform numbers results across runs for the random values generated in the code

I = []  # This is used to store the photons generated at each location in each turn
X = []  # This stores the position of every electron after each turn
V = []  # Stores the velocity of the electron after each turn


def electronCollision(u,xx,dx,kl):    # This function does the collision update as per the question    
    u[kl] = 0.0
    xx[kl] = xx[kl] - dx[kl]*np.random.rand(1)[0]


def electronCollisionModified(u,xx,dx,kl):   # This function does the collision update more accurately taking a random distribution of time 
    t = np.random.rand(1)[0]
    xx[kl] = xx[kl] - dx[kl]
    u[kl] = u[kl] - 1.0
    xx[kl] = xx[kl] + u[kl]*t + 0.5*t*t
    u[kl] = 0.0


ii = []
for k in range(1,nk):
    dx[ii] = u[ii] + 0.5 
    xx[ii] = xx[ii] + dx[ii]
    u[ii] = u[ii] + 1.0
    
    jj = np.where(xx > n)[0]
    dx[jj] = 0.0
    xx[jj] = 0.0
    u[jj] = 0.0
    
    kk = np.where( u >= u0 )[0]
    ll = np.where(np.random.rand(len(kk)) <= p)
    kl = kk[ll]

    electronCollisionModified(u, xx, dx, kl)
    I.extend(xx[kl].tolist())
    
    m = np.random.randn()*Msig + M     
    ll = np.where(xx == 0)[0]
    maxElec = min(len(ll),(int)(m))
    xx[ll[0:maxElec]] = 1.0
    u[ll[0:maxElec]] = 0.0

    ii = np.where(xx > 0)[0]
    X.extend(xx[ii].tolist())
    V.extend(u[ii].tolist())



# A 1 row 2 column subplot grid is declared
fig, axes = plt.subplots(1, 2, figsize=(20, 7)) 

# Population plot for electron density
axes[0].hist(X,histtype='bar',range=(10,n), bins=np.arange(1,n,n/100),ec='black',alpha=0.5) 
axes[0].set_title('Population Plot')
axes[0].set_xlabel('x')
axes[0].set_ylabel('No. of electrons')

# Population plot for intensity of emitted light
axes[1].hist(I,histtype='bar',range=(10,n), bins=np.arange(1,n,n/100),ec='black',alpha=0.5)
axes[1].set_title('Intensity Plot')
axes[1].set_xlabel('x')
axes[1].set_ylabel('No. of photons emitted ($\propto$ Intensity)')

plt.show()



plt.plot(X,V,'x')
plt.title('Electron phase space')
plt.xlabel('x')
plt.ylabel('velocity')
plt.show()



bins = plt.hist(I,bins=np.arange(1,n,n/100))[1]    # Bin positions are obtained
count = plt.hist(I,bins=np.arange(1,n,n/100))[0]   # Population counts obtained
xpos = 0.5*(bins[0:-1] + bins[1:])     # As no. of end-points of bins would be 1 more than actual no. of bins, the mean of bin end-points are used to get population of count a particular bin
df = pd.DataFrame()   # A pandas dataframe is initialized to do the tabular plotting of values.
df['Intensity'] = xpos
df['Count'] = count
print (df)

