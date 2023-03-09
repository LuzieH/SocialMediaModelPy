from src.SocialMediaModelPy import abm
import numpy as np

#paths for saving images and gif frames
imgpath = "img"
framespath = "img/frames" 

# parameters
N = 250 # number of individuals
timesteps = 1000 # time steps to simulate with a stepsize of dt ##350
a = 1 #0.5 ##1.5
b = 0. ##
#c = 0.5
theta = 2.9 # repulsion: 2.5, 2, 1, 0.5 attraction: 3 , interesting 2.9
seed = 0 # seed for random number generator
sigma = 0.1

# sample initial condition
x0,y0,z0,A,B,C0,D = abm.initialcondition(N, seed=seed)

# instantiate
ops = abm.opinions(x0, y0, z0, A, B, C0,D, b=b, a=a, theta=theta,sigma=sigma) 

# evolve model
xs,ys,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

# plot a snapshot
# ops.plotsnapshot(xs[-1],ys[-1],zs[-1],B,Cs[-1],save=True,path=imgpath)

# make gif
ops.makegif(xs,ys,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)

""" 

a_arr = np.linspace(0,1,1)
theta_arr = np.array([0.5])#, 1.0, 1.5, 2.0])
params_sensitivity = {"a": a_arr, "theta": theta_arr}

for param_key in params_sensitivity:
    for param in params_sensitivity[param_key]:
        #instantiate model with initial condition and parameters
        if param_key == "a":
            ops = abm.opinions(x0, y0, z0, A, B, C0,D, b=b, theta=theta, a=param) #c=c,
        elif param_key == "theta":
            ops = abm.opinions(x0, y0, z0, A, B, C0,D, b=b, a=a, theta=param) #c=c,
            break ##

        #evolve model
        xs,ys,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

        # plot a snapshot
        ops.plotsnapshot(xs[-1],ys[-1],zs[-1],B,Cs[-1],save=True,path=imgpath)

        # make gif
        ops.makegif(xs,ys,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)
        break """