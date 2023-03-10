from src.SocialMediaModelPy import abm
import numpy as np
#from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import matplotlib.pyplot as plt

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
    return [xs,zs] #,Cs #ys,


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
    num_simulations = 3 ##
    seeds = np.arange(num_simulations) # seed for random number generator
    stop_time_points = [250, 500] ##

    a_arr = np.linspace(0.1,1,5)
    theta_arr = np.array([0.5, 1.0, 1.5, 2.0])
    params_sensitivity = {"a": a_arr, "theta": theta_arr}

    for param_key in params_sensitivity:
        len_param = len(params_sensitivity[param_key])
        num_repeat = len_param * num_simulations

        abm_output_init = np.array(abm.initialcondition(N, seed=seeds[0]))
        for i in range(num_simulations):
            x0,y0,z0,A,B,C0,D = abm.initialcondition(N, seed=seeds[i])
            if i == 0:
                items_block = np.array([[x0,y0,z0,A,B,C0,D]] * len_param)
            else:
                items_block = np.concatenate([items_block, np.array([[x0,y0,z0,A,B,C0,D]] * len_param)], axis=0)

            if i == 0:
                domain = abm.opinions(x0, y0, z0, A, B, C0, D=D).domain
        
        if param_key == "a":
            items = np.concatenate(
                [
                    items_block, np.array([[b, theta]] * num_repeat), 
                    np.array([params_sensitivity[param_key]]*num_simulations).reshape(len_param*num_simulations,1), #np.array([params_sensitivity[param_key].T][:,None] * num_simulations),
                    np.array([[timesteps]] * num_repeat),
                    np.array(list(seeds) * len_param)[:,None]
                ], axis=1)
              
        elif param_key == "theta":
            items = np.concatenate(
                [
                    items_block, np.array([[b]] * num_repeat), 
                    np.array([params_sensitivity[param_key]] * num_simulations).reshape(len_param*num_simulations,1), #np.array([params_sensitivity[param_key].T][:,None] * num_simulations),
                    np.array([[a]] * num_repeat),
                    np.array([[timesteps]] * num_repeat),
                    np.array(list(seeds) * len_param)[:,None]
                ], axis=1)

        pool = multiprocessing.Pool(processes=len_param)
        results = pool.map(run_model, items) ##abm.opinions, run_model
        pool.close()

        xs = np.array(results)[:,0,:].reshape((num_simulations, len_param, timesteps+1))
        zs = np.array(results)[:,1,:].reshape((num_simulations, len_param, timesteps+1))

        for time_point in stop_time_points:
            xs_arr = np.array([xs[i,0,time_point] for i in range(num_simulations)])
            zs_arr = np.array([zs[i,0,time_point] for i in range(num_simulations)])
            
            plt.figure(figsize=(8,8))
            plt.hist2d(np.sum(xs_arr, axis=0)[:,0], np.sum(xs_arr, axis=0)[:,1], bins = 20, range = domain)
            plt.savefig("histogram_x_time_point_{0}.png".format(time_point))

            plt.hist2d(np.sum(zs_arr, axis=0)[:,0], np.sum(zs_arr, axis=0)[:,1], bins = 20, range = domain)
            plt.savefig("histogram_z_time_point_{0}.png".format(time_point))
    stop = time.time()
    print(stop-start)