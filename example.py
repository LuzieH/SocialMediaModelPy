from src.SocialMediaModelPy import abm
import numpy as np


#paths for saving images and gif frames
imgpath = "img"
framespath = "img/frames"

# parameters
N = 250 # number of individuals
timesteps = 500 # time steps to simulate with a stepsize of dt ##350
a = 0.1 
theta_ind = 1.5
theta_inf = 1.5
L = 4 # number of influencers
seed=0
level_off=False

x0,z0,A,C0,D = abm.initialcondition(N, L=4, seed=seed)

ops = abm.opinions(x0, z0, A, C0, D=D, theta_ind=theta_ind, theta_inf=theta_inf, a=a,level_off=level_off) 

xs,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

# make gif
ops.makegif(xs,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)