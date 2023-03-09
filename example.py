from src.SocialMediaModelPy import abm
import numpy as np

#paths for saving images and gif frames
imgpath = "img"
framespath = "img/frames" 

# parameters
N = 250 # number of individuals
timesteps = 1000 # time steps to simulate with a stepsize of dt ##350
a = 0.5 
theta_inf = 2.9 # repulsion: 2.5, 2, 1, 0.5 attraction: 3 , interesting 2.9
theta_ind = 2.9
seed = 0 # seed for random number generator
sigma = 0.1

# sample initial condition
x0,z0,A,C0,D = abm.initialcondition(N, L=4, seed=seed)

# instantiate
ops = abm.opinions(x0, z0, A, C0,D, a=a, theta_inf=theta_inf,theta_ind= theta_ind,sigma=sigma) 

# evolve model
xs,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

# plot a snapshot
# ops.plotsnapshot(xs[-1],ys[-1],zs[-1],B,Cs[-1],save=True,path=imgpath)

# make gif
ops.makegif(xs,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)
