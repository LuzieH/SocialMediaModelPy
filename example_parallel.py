from src.SocialMediaModelPy import abm
import numpy as np
#from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

# class opinions_model_a:
#     def __init__(self, x0, y0, z0, A, B, C0, D, b, theta):
#         self.x0 = x0
#         self.y0 = y0
#         self.z0 = z0
#         self.A = A
#         self.B = B
#         self.C0 = C0
#         self.D = D
#         self.b = b
#         self.theta = theta

#     #def run_model(self, a):
#     #    return abm.opinions(self.x0, self.y0, self.z0, self.A, self.B, self.C0, D=self.D, b=self.b, theta=self.theta, a=a)

def run_model(args):
    x0, y0, z0, A, B, C0, D, b, theta, a, timesteps, seed = args
    ops = abm.opinions(x0, y0, z0, A, B, C0, D=D, b=b, theta=theta, a=a)

    xs,ys,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

    # plot a snapshot
    #ops.plotsnapshot(xs[-1],ys[-1],zs[-1],B,Cs[-1],save=True,path=imgpath)

    # make gif
    #ops.makegif(xs,ys,zs,Cs,stepsize=10,gifpath=imgpath, framespath=framespath)
    return xs,ys,zs,Cs


if __name__ == "__main__":
    start = time.time()

    #paths for saving images and gif frames
    imgpath = "img"
    framespath = "img/frames" 

    # parameters
    N = 250 # number of individuals
    timesteps = 500 # time steps to simulate with a stepsize of dt ##350
    a = 0.5 ##1.5
    b = 0. ##
    ##c = 0.5
    theta = 1.5
    seed = 1 # seed for random number generator
    num_simulations = 2 ##

    # sample initial condition
    x0,y0,z0,A,B,C0,D = abm.initialcondition(N, seed=seed)

    a_arr = np.linspace(0.1,1,5)
    theta_arr = np.array([0.5, 1.0, 1.5, 2.0])
    params_sensitivity = {"a": a_arr, "theta": theta_arr}

    #max_workers = 61 # max_workers must be less than or equal to 61 (Windows)
    for param_key in params_sensitivity:
        #instantiate model with initial condition and parameters
        if param_key == "a":
            len_param = len(params_sensitivity[param_key])
            num_repeat = len_param * num_simulations
            items = np.concatenate([np.array([[x0,y0,z0, A, B, C0, D, b, theta]] * num_repeat),
                np.array([params_sensitivity[param_key]]*num_simulations).reshape(len_param*num_simulations,1), #np.array([params_sensitivity[param_key].T][:,None] * num_simulations),
                np.array([[timesteps, seed]] * num_repeat)], axis=1)
              
        elif param_key == "theta":
            len_param = len(params_sensitivity[param_key])
            num_repeat = len_param * num_simulations
            items = np.concatenate([np.array([[x0,y0,z0, A, B, C0, D, b]] * num_repeat),
                    np.array([params_sensitivity[param_key]]*num_simulations).reshape(len_param*num_simulations,1), #[params_sensitivity[param_key][:,None]],
                    np.array([[a]] * num_repeat),
                    np.array([[timesteps, seed]] * num_repeat)], axis=1)

        #with ProcessPoolExecutor(max_workers) as executor:
        pool = multiprocessing.Pool(processes=len_param)
        results = pool.map(run_model, items) ##abm.opinions, run_model
        pool.close()
        #print(results)
    stop = time.time()
    print(stop-start)