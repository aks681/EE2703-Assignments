
# coding: utf-8

# # Assignment 4
# ### Akshay Anand (EE16B046)

# Libraries that are used are declared

# In[124]:


import numpy as np
import math
from scipy.special import jv
from numpy.linalg import lstsq
import matplotlib.pyplot as plt


# Function for finding bessel function value declared

# In[125]:


def J(x,v):
    return jv(v,x) #in-built bessel function in scipy.special


# Vector and its corresponding function value initialised

# In[126]:


h = 0.5  #step-size
np.random.seed(2)  # A seed given so that uniform noise is generated across step-size values
x = np.linspace(0.0, 20.0, (20.0)/h + 1) # vector initialised
y = J(x,1)


# The function is plotted from 0 to 20 (at a lower step-size so as to obtain a smooth graph.

# In[127]:


x_plot = np.linspace(0.0,20.0,101)
plt.plot(x_plot,J(x_plot,1))
plt.grid()
plt.title('Plot of $J_1(x)$')
plt.ylabel('$J_1(x)$')
plt.xlabel('$x$')
plt.show()


# The function for calculating $\nu$ is defined
# This is done by solving the approximate matrix equation using least squares estimation.
# 
# Then $\phi$ is calculated as $\cos^{-1}(\frac{A}{\sqrt{A^2 + B^2}})$ in the equation $A\cos(x_i) + B\sin(x_i) \approx J_1(x_i)$ in Qn (b) 
# 
# or $A\frac{\cos(x_i)}{\sqrt{x_i}} + B\frac{\sin(x_i)}{\sqrt{x_i}} \approx J_1(x_i)$ in Qn (c)

# In[128]:


def calcnu(x,x0,eps,model):
    i = np.where(x==x0)[0][0] # index in x corresponding to x0 calculated
    x_new = x[i:len(x)] # sub vector of x extracted based on starting index
    y_new = y[i:len(x)] # sub vector of y extracted based on starting index
    y_new = y_new + eps*np.random.randn(len(y_new))
    A = np.zeros((len(x_new),2)) # The 2D matrix is initialised and assigned values in the following line
    if (model == 'b'): # model 'b' corresponds to the equation in question (b)
        A[:,0]=np.cos(x_new)
        A[:,1]=np.sin(x_new)
    elif (model == 'c'): # model 'c' corresponds to the equation in question (c)
        A[:,0]=np.cos(x_new)/np.sqrt(x_new)
        A[:,1]=np.sin(x_new)/np.sqrt(x_new)
    c = lstsq(A,y_new)[0] # Values of A and B found as c[0] and c[1] respectively
    phi = math.acos(c[0]/(np.sqrt(c[0]*c[0] + c[1]*c[1]))) # phi calculated
    v = phi - (math.pi/4)
    v = v / (math.pi/2) # nu finally calculated and returned
    return v


# The above function is called for Qn (b), Qn(c) and Qn(c) with noise.

# In[129]:


x0 = np.linspace(0.5,18.0,(18.0-0.5)/h + 1)
nu_b = []
nu_c_no_noise = []
nu_c_with_noise = []
for i in x0:
    nu_b.append(calcnu(x,i,0,'b'))
    nu_c_no_noise.append(calcnu(x,i,0,'c'))
    nu_c_with_noise.append(calcnu(x,i,0.01,'c'))
print ("The maximum error between the calculated nu values with and without noise is: ",max(np.absolute(np.array(nu_c_no_noise)-np.array(nu_c_with_noise))))


# The various $\nu$ values obtained are plotted against $x_0$

# In[130]:


plt.plot(x0, nu_b, 'bo', markeredgecolor='black')
plt.plot(x0, nu_c_no_noise, 'go', markeredgecolor='black')
plt.plot(x0, nu_c_with_noise, 'ro', markeredgecolor='black')
plt.legend(('$\epsilon$ = 0, model (b)','$\epsilon$ = 0, model (c)','$\epsilon$ = 1.0e-02, model (c)'))
plt.grid()
plt.title(r'Plot of $\nu$ vs $x_0$')
plt.xlabel('$x_0$')
plt.ylabel(r'$\nu$')
plt.show()

