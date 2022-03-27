# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 00:01:53 2022

@author: Bing2Na
"""
import numpy as np

def unit(x):
    A = np.array([int(x[0]),int(x[1]),int(x[2])])
    return x/np.linalg.norm(A)
    
def Mo(x):
    return np.linalg.norm(x)
def D_R(x):
    return x*np.pi/180
def R_D(x):
    return x*180/np.pi
from sympy import *
x = Symbol('x')
y = Symbol('y')

def cosABR(a,b,r):
    return solve(x**2+a**2-b**2-2*x*a*np.cos(r),x)


#######################################################
#第4.1题

k1 = 211 # N/mm
k2 = 110 # N/mm
L = 200 # mm
theta = 23 # degrees
P = 19 # N

#######################################

M = P*(1/2)
F2 = k2*x
x1 = L*np.sin(D_R(theta))
y1 = L*np.cos(D_R(theta))
y2 = y1+x
L2 = (y2**2+x1**2)**0.5
x1 = L2-L
F1 = k1*x1*np.cos(D_R(theta))
ans = solve(F1+F2-M,x)
#print(ans[1])  #Answer


#######################################################
#第4.2题

h = 110 # mm
b = 102 # mm
d = 13 # mm
E1 = 187 # MPa
E2 = 210 # MPa
W = 87 # N

#######################################
x = 0.1
xc = x
xd = 2*x
lc = ((h)**2+b**2)**0.5
ld = ((h)**2+(b*2)**2)**0.5
lc1 = ((h+xc)**2+b**2)**0.5
ld1 = ((h+xd)**2+(b*2)**2)**0.5
dlc = lc1-lc
dld = ld1-ld
R = dlc/dld
Ac = (d/2)**2*np.pi
f1 = lc/(E1*Ac)
f2 = ld/(E2*Ac)
F1_F2 = R*f2/f1
M = b*W/2+b*2*W/2
sin1 = h/lc
sin2 = h/ld
F1 = solve(b*y*sin1+(y/F1_F2)*sin2*2*b-M,y)[0]
F2 = F1/F1_F2
#print('Ratio is',R,'\nf1 =',f1,'\nf2 =',f2,'\nF1/F2 = ',F1_F2,'\nsigma_1 is',F1/Ac,'\nsigma_2 is',F2/Ac)


#######################################################
#第4.3题

E1 = 70 # GPa
E2 = 200 # GPa
d = 28 # mm
D = 42 # mm
L = 742 # mm
P = 93 # kN

#######################################

Acr = (d/2)**2*np.pi
Act = (D/2)**2*np.pi - Acr
f1 = L/(Acr*E1)
f2 = L/(Act*E2)
F = P*1000/(f1+f2)*f1
#print(-F/Act)  #Answer

#######################################################
#第4.4题

k1 = 800000 # kN/m
k2 = 300000 # kN/m
P = 73 # kN

#######################################

d1 = P/2/k2
d2 = d1+P/k1
#print(d1*1000,d2*1000)

#######################################################
#第4.5题

E1 = 71 # GPa
E2 = 207 # GPa
A = 1714 # mm^2
L = 728 # mm
deltaM = 0.4 # mm

#######################################

L1 = L
L2 = L+deltaM
s1 = E1*L1
s2 = E2*L2
delta1 = deltaM*s2/(s1+s2)/L1
delta2 = deltaM*s1/(s1+s2)/L2
#print('d1 is',-delta1*1000,'\nd2 is',-delta2*1000)  #Answer

#######################################################
#第4.6题

deltaM = 2 # mm
k1 = 44000 # N/mm
k2 = 13000 # N/mm
L = 2200 # mm

#######################################

L1 = L-deltaM
L2 = L
delta1 = deltaM*k2/(k1+k2)/L1
delta2 = deltaM*k1/(k1+k2)/L2
#print('d1 is',delta1*1000,'\nd2 is',-delta2*1000)  #Answer

#######################################################
#第4.7题

delta = 0.01 # in
L = 13 # in
E = 31000 # ksi
alpha = 0.000007 # /degrees F
A = 2 # in^2
dT = -30 # degrees F

#######################################

L0 = L+delta
L1 = L0+alpha*dT*L
strain = (L1-L)/L
Pn = strain*E
#print(-Pn)

#######################################################
#第4.8题

dBrass = 28 # mm
dSteel = 17 # mm
Ls = 758 # mm
Lb = 765 # mm
Es = 184 # GPa
Eb = 87 # GPa
P = 35 # kN

#######################################

Acb = (dBrass/2)**2*np.pi
Acs = (dSteel/2)**2*np.pi
sb = 2*Eb*Acb/Lb  #没错这是stiffness of Brass
ss=  Es*Acs/Ls
Pnb = 1000*-P*sb/(sb+ss)/(2*Acb)
Pns = 1000*P*ss/(sb+ss)/Acs
#print(Pnb,Pnb,Pns)  #如果P向左
#print(-Pnb,-Pnb,-Pns)  #如果P向右










