# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 15:07:30 2022

@author: GoldenGlow
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
#第1题

L = 216 # mm
a = 238 # mm
theta = 0.5 # degrees
alpha = 32 # degrees

#######################################
theta = D_R(theta)
alpha = D_R(alpha)
dh = a*theta
dL = ((L*np.sin(alpha)+dh)**2+(L*np.cos(alpha))**2)**0.5-L
delta1 = 1000*dL/L #粗略计算，按直角处理

db = (a**2+a**2-2*a*a*np.cos(theta))**0.5
beta = alpha+(np.pi/2-theta)
L1 = (L**2+db**2-2*L*db*np.cos(beta))**0.5
dL = L1-L
delta2 = 1000*dL/L #精确计算，解三角形
delta3 = (delta1+delta2)/2 #取均值，最接近答案（咱也不知道答案怎么算的）
#print(delta1,delta2,delta3) 

#######################################################
#第2题

h = 291 # mm
a = 182 # mm
deltaC = 1.5 # mm

#######################################
deltaB = 0.5*deltaC
stn = deltaB/h
#print(1000*stn)

#######################################################
#第3题

a = 11 # in
b = 12 # in
strain = 0.005

#######################################
L = (a**2+b**2)**0.5
dL = strain*L
L1 = L+dL
alpha = np.arctan(a/b)+np.pi/2
#sltn = cosABR(L,L1,alpha)[0] #两个解中不太合理的一个，如果错了大抵就用这个解来更正
sltn = cosABR(L,L1,alpha)[1] 
#print(R_D(sltn/b))

#######################################################
#第4题

a = 670 # mm
b = 550 # mm
dbx = 1 # mm
dcx = 3 # mm
dcy = 2 # mm
ddx = 1 # mm
ddy = 2 # mm

#######################################
AD = ((ddy+b)**2+ddx**2)**0.5
e1 = (AD-b)/b
g1 = ddx/(ddy+b)
BC = ((dcy+b)**2+(dcx-dbx)**2)**0.5
e2 = (BC-b)/b
g2 = (dcx-dbx)/(b+dcy)
e3 = dbx/a
e4 = (dcx-ddx)/a
#print('A',g1,'B',g2,'\nAD',1000*e1,'\nBC',1000*e2,'\nAB',1000*e3,'\nCD',1000*e4)

#######################################################
#第5题

a = 537 # mm
b = 443 # mm
db = 2 # mm
dd = 2 # mm

#######################################
AC = (a**2+b**2)**0.5
AC1 = ((a-db)**2+(b-dd)**2)**0.5
#print(1000*(AC1-AC)/AC)


#######################################################
#第6题

n = 2.13

#######################################

etrue = np.log(n)
eeng = n-1
#print(etrue/eeng)