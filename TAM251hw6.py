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
#HW6.1

R1 = 192 # mm
R2 = 48 # mm
d = 23 # mm
TW = 4461 # N.mm

#######################################
R1*=1e-3; R2*=1e-3; d*=1e-3; TW*=1e-3
TE = TW*R1**2/R2**2
tao = TE*d/2/getJ(d/2)
#print(tao*1e-6)

#######################################################
#HW6.2

d = 30 # mm
R1 = 100 # mm
R2 = 50 # mm
G = 75 # GPa
L = 120 # mm
TE = 483 # N.m

#######################################
R1*=1e-3; R2*=1e-3; d*=1e-3; L*=1e-3; G*=1e9
T1 = TE*R2**2/R1**2
T2 = TE*R2/R1
T3 = TE
k = L/getJ(d/2)/G
phi1 = k*T1; phi2 = k*T2; phi3 = k*T3
phi = phi1*R2**2/R1**2+phi2*R2/R1+phi3
#print(phi)

#######################################################
#HW6.3

di = 27 # mm
TB = 1516 # N.m
TC = 535 # N.m
G = 101 # MPa
L = 1148 # mm
tauMax = 12 # MPa

#######################################
di*=1e-3; L*=1e-3; G*=1e6; tauMax*=1e6
T1 = TB-TC
T2 = -TC
r = di/2
minR = solve(abs(T1)*x/(getJ(x))-tauMax,x)
D = 2*minR[0]
phi1 = T1*L/G/getJ(D/2)
phi2 = T2*L/G/(getJ(D/2)-getJ(r))
phi = abs(phi1-phi2)
#print(T1,T2,D*1000,phi1+phi2)

#######################################################
#HW6.4

d = 45 # mm
L = 195 # mm
G = 54 # GPa
TB = 1497 # N.m

#######################################
d*=1e-3; L*=1e-3; G*=1e9
R = d/2
r = d/4
k1 = (L/G/getJ(R))**-1
k2 = (L/G/(getJ(R)-getJ(r)))**-1
phi = TB/(k1+k2)
#print(phi)

#######################################################
#HW6.5

L1 = 245 # mm
L2 = 643 # mm
L3 = 251 # mm
d1 = 28 # mm
d2 = 46 # mm
G = 63 # GPa
Tb = 1342 # N.m
Tc = 1304 # N.m

#######################################

L1*=1e-3; L2*=1e-3; L3*=1e-3;d1*=1e-3; d2*=1e-3; G*=1e9
#Tb*=-1 #if Tb to the left
#Tc*=-1 #if Tc to the left
k = np.array([(L1/G/getJ(d1/2))**-1,(L2/G/getJ(d2/2))**-1,(L3/G/(getJ(d2/2)-getJ(d1/2)))**-1])
#solve x as the torsion on the left end
solu = solve(x/k[0]+(Tb+x)/k[1]+(Tb+Tc+x)/k[2],x)
T1 = float(solu[0])
T2 = Tb+T1
T3 = Tb+Tc+T1
tau1 = abs(-T1*d1/2/getJ(d1/2))
tau2 = abs(-T2*d2/2/getJ(d2/2))
tau3 = abs(-T3*d2/2/(getJ(d2/2)-getJ(d1/2)))
#print('AB:',tau1*1e-6,'\nBC:',tau2*1e-6,'\nCD:',tau3*1e-6)

#######################################################
#HW6.6

L = 45 # in
Ga = 5000 # ksi
Gs = 10000 # ksi
da = 5 # in
ds = 4 # in
Tb = 2 # kip.in

#######################################

Ga*=1e3; Gs*=1e3; Tb*=1e3
R = da/2
r = ds/2
ks = (L/Gs/getJ(r))**-1
ka = (L/Ga/(getJ(R)-getJ(r)))**-1
phi = Tb/(ka+ks)
#print(phi)


#######################################################
#HW6.7

L = 45 # in
Ga = 5000 # ksi
Gs = 10000 # ksi
da = 5 # in
ds = 4 # in
Tb = 2 # kip.in

#######################################

Ga*=1e3; Gs*=1e3; Tb*=1e3
R = da/2
r = ds/2
ks = (L/Gs/getJ(r))**-1
ka = (L/Ga/(getJ(R)-getJ(r)))**-1
Ts = Tb*ks/(ka+ks)
Ta = Tb*ka/(ka+ks)
taus = Ts*r/getJ(r)
taua = Ta*R/(getJ(R)-getJ(r))
#print('Steel:',taus*1e-3,'\nAlum:',taua*1e-3)


"""
Finished 2022.3.28
Last Editted 2022.3.28
"""