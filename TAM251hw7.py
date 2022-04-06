"""
@author: Bing2Na
"""


#Assumptions and importing
from sympy import *
x = Symbol('x')
y = Symbol('y')
import numpy as np

#Some notes for functions, which gonna be improved by you, who using me

#unit({a sympy Matrix})
    #Convert a Sympy Matrix into a numpy array, get norm of a vector
def unit(x):
    A = np.array([int(x[0]),int(x[1]),int(x[2])])
    return x/np.linalg.norm(A)

#Mo({an array}): Short for np.linalg.norm()
    #get norm of a numpy array vector
#Mag({an Matrix}):
    #get magnitude of a 1*3 Matrix
def Mo(x):
    return np.linalg.norm(x)
def Mag(x):
    return (x[0]**2+x[1]**2+x[2]**2)**0.5

#D_R({angle in degrees}) and R_D({angle in radians})
    #convert "degree" to "radians" or reversed
def D_R(x):
    return x*np.pi/180
def R_D(x):
    return x*180/np.pi

#cosABR({the side next to the angle, the side face to the angle, the angle in radians})
    #solve the length of one side next to the given angle by Law of cosines,
    #called "ASS" condition in solving a triangle,
    #for the easier one "SAS", try yourself(doge)
def cosABR(a,b,r):
    return solve(x**2+a**2-b**2-2*x*a*np.cos(r),x)
#getJ({radius})
    #To get the polar moment of inertia of a solid shaft
def getJ(x):
    return np.pi*x**4/2


#######################################################
#HW7.7

h = 153 # mm
b = 248 # mm
t1 = 20 # mm
t2 = 22 # mm
Mz = 1272 # N.m

#######################################

H = h+t2*2
B = b+t1*2
Iz = B*H**3/12-b*h**3/12
sigma = Mz/Iz*(H/2)
#print(t2+h/2,Iz*1e-6,sigma*1e3)

#######################################################
#HW7.8

t = 18 # mm
Mz = 1751 # N.m

'''
Tensive stress or Compressive stress? (1 for Tensile, -1 for Compressive)
'''
Situation = 1

#######################################

H = 10*t
B = 4*t
h = 4*t
b = 2*t
C = (H*B*H/2+(-h*b*3*t))/(H*B-h*b)
ybar = C-3*t
L1 = C-H/2
I1 = B*H**3/12+B*H*L1**2
I2 = b*h**3/12+b*h*ybar**2
Iz = I1-I2
if Mz*Situation < 0:
    sigma = (10*t-C)*Mz/Iz
else:
    sigma = C*Mz/Iz
#print(ybar,Iz*1e-6,1e3*abs(sigma))

#######################################################
#HW7.9

R = 25 # mm
Mz = -1787 # N.m

'''
Tensive stress or Compressive stress? (1 for Tensile, -1 for Compressive)
'''
Situation = 1

#######################################

b = np.pi*R
h = 4*R
C = R*b*2.5*R/(R*b+3*np.pi*R**2)
ybar = C
I1 = 61*np.pi*R**4/12+3*np.pi*R**2*C**2
I2 = R**3*np.pi*R/12+np.pi*R**2*(2.5*R-C)**2
Iz = I1+I2
if Mz*Situation < 0:
    sigma = (3*R-C)*Mz/Iz
else:
    sigma = (2*R+C)*Mz/Iz
#print(C,Iz*1e-6,1e3*abs(sigma))

#######################################################
#HW7.9

t = 20 # mm
a = 58 # mm

'''
hollow at upper half or lower? (1 for Upper, -1 for Lower)
'''
Situation = -1

#######################################

l = a+t
H = 12*t
B =  8*t
h = 2*t
b = 6*t
C = -h*b*l/(H*B-h*b)
#print(Situation*C)

#######################################################
#HW7.9

t = 16 # mm
a = 16 # mm
r = 48 # mm
ybar = 5.2542 # mm

#######################################
H = 14*t
B = 8*t
d = 4*r/(3*np.pi)
l = d+a
I1 = H**3*B/12 + H*B*(ybar**2)
I2 = (np.pi/8-8/(9*np.pi))*r**4 + np.pi*r**2/2*(l+abs(ybar))**2
Iz = I1-I2
#print(Iz*1e-6)


"""
Finished 2022.4.6
Last Editted 2022.4.6
"""




