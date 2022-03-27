# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:31:26 2022

@author: 31857
"""
import numpy as np
def Mo(x):
    return np.linalg.norm(x)
def D_R(x):
    return x*np.pi/180
def R_D(x):
    return x*180/np.pi
from sympy import *
x = Symbol('x')
y = Symbol('y')

#######################################################
#第3.7题

Lo = 200 # mm
d = 27 # mm
P = 133 # kN
delta = 0.1 # mm
sy = 350 # MPa
G = 200 # GPa

#######################################


e = delta/Lo #y方向的strain
P = P*1000
d = d/1000
A = np.pi*(d/2)**2 #截面面积
stress = P/A #y方向的压强
E = stress/e #y方向的模量
print(E/1000000000)
sy = 1000000*sy
G = 1000000000*G
v = E/(2*G)-1 #G = E/(2*(v+1))
print(v)
dx = abs(-e*v*d)
print(dx*1000)

#######################################################
#第3.8题


G = 3000 # ksi
P = 636 # kips
Lo = 10 # in
b = 7 # in
t = 2 # in

#######################################
tao = P/Lo/b
print(tao)
print(t*tao/G)

#######################################################
#第3.9题

P = 2 # kN
a = 668 # mm
b = 522 # mm
L = 1045 # mm
E = 200 # GPa
Ac = 934 # mm^2
theta = 54 # degrees

#######################################

Fn = P*L/a
F = Fn/np.sin(D_R(theta))
s = F/Ac
delta = b*s/E
db = delta/np.sin(D_R(theta))
dc = L*db/a
print(dc)


L = 2211 # mm
Ac = 3665 # mm^2
E = 210 # GPa
P = 466 # kN
theta = 40 # degrees
cs = np.cos(D_R(theta))
sn = np.sin(D_R(theta))
#F = solve([x*cs-y*cs-P,x*sn+y*sn-P],[x,y])[x] #P朝右
F = solve([x*cs-y*cs-P,x*sn+y*sn-P],[x,y])[y] #P朝左
Fn = F*np.cos((D_R(theta)))
delta = Fn/Ac/E
print(delta*L)

P = 78 # kN
d = 22 # mm
E = 85 # GPa
L = 685 # mm
a = 205 # mm
b = 336 # mm
alpha = 38 # degrees
#Fn = P*b/(a+b)/np.cos(D_R(alpha)) #P朝左
Fn = -P*b/(a+b)/np.cos(D_R(alpha)) #P朝右
print(Fn)
A = np.pi*((d/2)**2)
sn = Fn/A
delta = L*sn/E
print(delta)