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
def Mo(x):
    return np.linalg.norm(x)

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
#HW5.1

L = 1651 # mm
r = 20 # mm
G = 59 # GPa
tu = 53 # MPa
FS = 2

#######################################
r = r/1000
tu = tu*1000000
L = L/1000
G = G*1000000000
Tm = tu*getJ(r)/r/FS
phi = Tm*L/getJ(r)/G
#print(phi)

#######################################################
#HW5.2

L = 1765 # mm
d = 29 # mm
kt = 4 # kN.m
T = 1813 # N.m

#######################################
kt = 1000*kt
d = d/1000
phi = T/kt
J = getJ(d/2)
L = L/1000
#print(1000*phi*d/2/L)

#######################################################
#HW5.3

L = 1660 # mm
di = 20 # mm
do = 36 # mm
G = 31 # GPa
T = 1627 # N.m

#######################################
L = L/1e3;di = di/1e3;do = do/1e3;G = G*1e9
J = getJ(do/2)-getJ(di/2)
phi = L*T/G/J
#print(1e3*phi*di/2/L)

#######################################################
#HW5.4

L1 = 324 # mm
L2 = 309 # mm
d1 = 71 # mm
d2 = 46 # mm
G1 = 39 # GPa
G2 = 41 # GPa
TB = 627 # N.m
TC = 541 # N.m

#######################################

L1 = L1/1e3;L2 = L2/1e3;d1 = d1/1e3;d2 = d2/1e3
G1 = G1*1e9;G2 = G2*1e9
phi2 = TC*L2/getJ(d2/2)/G2
phi1 = (TC-TB)*L1/getJ(d1/2)/G1
#print(phi1+phi2)

#######################################################
#HW5.5

L1 = 375 # mm
L2 = 320 # mm
d1 = 80 # mm
d2 = 45 # mm
G1 = 39 # GPa
G2 = 37 # GPa
TB = 957 # N.m
TC = 736 # N.m

#######################################

L1 = L1/1e3;L2 = L2/1e3;d1 = d1/1e3;d2 = d2/1e3
G1 = G1*1e9;G2 = G2*1e9
tao2 = TC*d2/2/getJ(d2/2)
tao1 = (TC-TB)*d1/2/getJ(d1/2)
#print(max(tao2,tao1)/1e6)

#######################################################
#HW5.6

d = 67 # mm
TB = 1279 # N.m
TC = 935 # N.m

#######################################

d = d/1e3
J1 = getJ(d/2)-getJ(d/4)
J2 = getJ(d/2)
tao2 = TC*d/2/J2
tao1 = (TC-TB)*d/2/J1
#print(max(abs(tao1),abs(tao2))/1e6)

#######################################################
#HW5.7

Rb = 109 # mm
Rc = 36 # mm
ft = 0.000002 # (1/N.m)
TD = 614 # N.m

#######################################

TB = TD*Rb/Rc
phi = TB*ft*Rb/Rc+TD*ft
#print(phi)

#######################################################
#HW5.8

Rb = 71 # mm
Rc = 34 # mm
d = 14 # mm
TD = 45 # N.m

#######################################

TB = TD*Rb/Rc
d = d/1e3
taoB = TB*d/2/getJ(d/2)
taoD = TD*d/2/getJ(d/2)
#print(max(abs(taoB),abs(taoD))/1e6)

#######################################################
#HW5.9

a = 1201 # mm
b = 1533 # mm
d = 30 # mm
G = 67 # GPa
tall = 20 # MPa

#######################################

a = a/1e3;b = b/1e3;d = d/1e3;G = G*1e9;tall = tall*1e6
T = tall*getJ(d/2)/d*2
P = T/a
phi = T*b/getJ(d/2)/G
#print(P,phi)