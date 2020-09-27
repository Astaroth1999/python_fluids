# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 23:08:51 2020

@author: Guillem
"""
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
from matplotlib import animation
from time import time

plt.ioff()

@jit(cache = True)
def velocity(vel0,diff,p,D):
        
    """Esta función calcula la variacion del campo de velocidades en función del tiempo:
        
    Se resuelven las EDP's de Navier-Stokes, mediante la discretización de estas
    Las condiciones de contorno se mantienen siempre constantes:
        0 para la velocidad
        1 para la densidad
    """
    U, V = vel0
    dx, dy, dt = diff
    row, col = U.shape
    for j in range(1, row-1):
        for i in range(1, col-1):
                
            dudx   = 1/dx*(U[i,j]-U[i-1,j])
            dudy   = 1/dy*(U[i,j]-U[i,j-1])
            dvdx   = 1/dx*(V[i,j]-V[i-1,j])
            dvdy   = 1/dy*(V[i,j]-V[i,j-1])
            d2udx2 = 1/dx**2*(U[i+1,j]-2*U[i,j]+U[i-1,j])
            d2udy2 = 1/dy**2*(U[i,j+1]-2*U[i,j]+U[i,j-1])
            d2vdx2 = 1/dx**2*(V[i+1,j]-2*V[i,j]+V[i-1,j])
            d2vdy2 = 1/dy**2*(V[i,j+1]-2*V[i,j]+V[i,j-1])
            
            dudt   = D*(d2udx2+d2udy2) - U[i,j]*(dudx+dudy)
            dvdt   = D*(d2vdx2+d2vdy2) - V[i,j]*(dvdx+dvdy)
            dpdt   = -p[i,j]*(dudx+dudy)
        
            U[i,j] += dt*dudt 
            V[i,j] += dt*dvdt 
            p[i,j] += dt*dpdt 
        
    U[0, :] = 0
    U[-1,:] = 0
    U[:, 0] = 0
    U[:,-1] = 0
    V[0, :] = 0
    V[-1,:] = 0
    V[:, 0] = 0
    V[:,-1] = 0
    p[0, :] = 1
    p[-1,:] = 1
    p[:, 0] = 1
    p[:,-1] = 1
    
    return U, V, p

@jit(cache = True)
def vorticity(vel0, diff, w):
    
    
    U, V = vel0
    dx, dy, dt = diff
    
    row, col = w.shape
    w = np.zeros((row,col))
    for j in range(1, row-1):        
        for i in range(1, col-1):    
            w[j,i] = (V[j,i] - V[j-1,i])/dx - (U[j,i] - U[j,i-1])/dy
    
    w[-1, :] = w[-2,:]		
    w[:, -1] = w[:,-2]
    return w

  


@jit
def circle(U,V,resolution,par):
  
    x, y = par
    for j in range(0, resolution):
        for i in range(0, resolution):
            if (x[i]-3)**2 +(y[j]-3)**2 <= 4:
                U[j,i] = 2
                V[j,i] = 2

    return [U, V]


def update_quiver(i,X,Y,p,D,w,vel0,diff,ax, velocity): 
    
    global Q, C, Cf
    ax.collections = []

    U, V, p = velocity(vel0,diff,p,D)
    w = vorticity(vel0, diff, w)
    #Q = ax.quiver(X[::2, ::2], Y[::2, ::2], U[::2, ::2], V[::2, ::2])
    C = ax.contourf(X, Y, w, alpha=0.5, cmap=plt.cm.winter)
    
    return  C #, Q,




# def main():
t0 = time()
D, n, resolution  = 0.01, 10000, 100

                    
Lx, Ly = 20, 20				                 			      
	      
x, y  = np.linspace(0, Lx, resolution), np.linspace(0, Ly, resolution)
row,col = resolution,resolution
U0,V0 = np.zeros((row,col)), np.zeros((row,col))
w, p = np.zeros((row,col)), np.ones((row,col))
par  = x,y
diff = Lx/(resolution-1), Ly/(resolution-1), 0.01
T = np.linspace(0,n*diff[2],n)

vel0 = circle(U0,V0,resolution,par)
fig, ax = plt.subplots(1,1)
X, Y = np.meshgrid(x, y)

C = ax.contourf(X, Y, w, alpha=0.5, cmap=plt.cm.winter)
#Q = ax.quiver(X[::2, ::2], Y[::2, ::2], U0[::2, ::2], V0[::2, ::2])

anim = animation.FuncAnimation(fig, update_quiver , fargs=(X,Y,p,D,w,vel0,
                                    diff,ax, velocity),frames = T,interval = 1, blit=False)

anim.save('prueba.mp4', writer = 'ffmpeg', fps = 120 )
tf = time() - t0

print(tf)
# if __name__ == "__main__":
#     main()