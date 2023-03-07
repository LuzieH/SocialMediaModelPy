from src.SocialMediaModelPy import abm

#paths for saving images and gif frames
imgpath = "img"
framespath = "img/frames" 

# parameters
N = 250 # number of individuals
timesteps = 350 # time steps to simulate with a stepsize of dt
a = 1. ##1.5
b = 0. ##
c = 1.
seed = 1 # seed for random number generator

# sample initial condition
x0,y0,z0,A,B,C0 = abm.init(N, seed=seed)

#instantiate model with initial condition and parameters
ops = abm.opinions(x0, y0, z0, A, B, C0, a=a, b=b, c=c)

#evolve model
xs,ys,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

# plot a snapshot
ops.plotsnapshot(xs[-1],ys[-1],zs[-1],B,Cs[-1],save=True,path=imgpath)

# make gif
ops.makegif(xs,ys,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)