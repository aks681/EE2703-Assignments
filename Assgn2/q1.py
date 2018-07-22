import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pandas as pd

h = 0.1
x = np.linspace(0.0, 5.0, num=(5.0/h) + 1)

# The function to calculate 1/(1+t^2)

def function(t):
	return 1/(1+t*t)


# The integral is calculated using trapezoidal integration method with step-size - h

def trapezoidalIntegral(h,a,b):
	x = np.linspace(0.0, 5.0, num = (5.0/h + 1))
	y = h*(np.cumsum(function(x)) - 0.5*(function(0) + function(x)))
	return y

# Plot the function 1/(1+t^2) vs t

plt.plot(x, function(x))
plt.title('Plot of $1/(1+t^{2})$')
plt.xlabel('t')
plt.ylabel('$1/(1+t^{2})$')
plt.show()

# Tabulate the values of tan^(-1) x vs the integral calculated using scipy.integrate.quad.

y = []
for i in x:
	y.append(quad(function, 0, i)[0])
df = pd.DataFrame() 		#df is a pandas DataFrame to tabulate the values
df['tan^(-1) x'] = np.arctan(x)
df['Integral with quad'] = y
df.to_csv('table.csv')

# Plot the above tabulated values against x

plt.plot(x, y, 'ro') 			#Plot the result of the integral
plt.plot(x, np.arctan(x), '#000000') 	#Plot the actual value of tan^(-1) x
plt.legend(('quad function','$tan^{-1} x$'))
plt.title('Integral plot using scipy.integrate.quad and actual value')
plt.xlabel('x')
plt.ylabel('$\int_0^x 1/(1+t^2) dt$')
plt.show()

# Plot the error between the above obtained values and the actual values of $tan^{-1} x$

plt.semilogy(x, abs(y - np.arctan(x)), 'ro')
plt.title('Error plot between scipy.integrate.quad and actual value')
plt.xlabel('x')
plt.ylabel('Error')
plt.show()

# Calculate the trapezoidal integral of the function

trapezoid = trapezoidalIntegral(0.1,0,5)
plt.title('Integral plot using scipy.integrate.quad, actual value and trapezoidal integration')
plt.xlabel('x')
plt.ylabel('$\int_0^x 1/(1+t^2) dt$')
plt.plot(x, np.arctan(x), 'g')
plt.plot(x, y, 'ro')
plt.plot(x, trapezoid, '+')
plt.legend(('$tan^{-1} x$', 'quad function', 'Trapezoidal method'))
plt.show()

# Calculate Actual Error ( difference between the values obtained through trapezoidal integration and
# the actual values of tan^(-1) x ) and the Estimated Error ( the difference between common values of function in
# 2 successive iterations of the step variable h )

estError = []
actError = []
h = 0.1
hList = []
maxError = 1
while maxError > 10**-8:
	trapezoid = trapezoidalIntegral(h,0,5)
	actError.append(max(abs(trapezoid - np.arctan(np.linspace(0, 5, num = (int)(5/h + 1))))))
	hList.append(h)
	h = h/2
	nextTrapezoid = trapezoidalIntegral(h,0,5)
	maxError = max(abs(trapezoid - nextTrapezoid[::2]))
	estError.append(maxError)

# Plot the above obtained Actual and Estimated errors against step-size h

plt.loglog(hList, actError, 'ro')
plt.loglog(hList, estError, '+')
plt.title('Error plot')
plt.xlabel('Step-size')
plt.ylabel('Error magnitude')
plt.legend(('Exact Error','Estimated Error'))
plt.show()
