from src.SocialMediaModelPy import abm
import numpy as np
import multiprocessing
import time
import matplotlib.pyplot as plt


def run_model(args):
    x0, z0, A, C0, D,theta_ind, theta_inf, a, timesteps, seed = args 
    ops = abm.opinions(x0, z0, A, C0, D=D, theta_ind=theta_ind, theta_inf=theta_inf, a=a) 

    xs,zs,Cs = ops.run(timesteps=timesteps, seed=seed)

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
    theta_ind = 1.5
    theta_inf = 0.5
    L = 4 # number of influencers

    num_simulations = 50 ##
    seeds = np.arange(num_simulations) # seed for random number generator
    stop_time_points = [250, 500] ##

    a_arr = np.linspace(0.1,1,3)
    theta_ind_arr = np.array([0.5, 1.5, 2.5]) #np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    theta_inf_arr = np.array([0.5, 1.5, 2.5])

    param_combinations = [len(a_arr), len(theta_ind_arr), len(theta_inf_arr)]

    params_sensitivity = {"a": a_arr, "theta_ind": theta_ind_arr, "theta_inf": theta_inf_arr}
    params_len = {"a": len(a_arr), "theta_ind": len(theta_ind_arr), "theta_inf": len(theta_inf_arr)}

    for param_key in params_sensitivity:
        len_param = len(params_sensitivity[param_key])
        num_repeat = len_param * num_simulations

        for i in range(num_simulations):
            x0, z0, A, C0, D = abm.initialcondition(N, seed=seeds[i])
            if i == 0:
                items_block = np.array([[x0,z0,A,C0,D]] * len_param)
            else:
                items_block = np.concatenate([items_block, np.array([[x0,z0,A,C0,D]] * len_param)], axis=0)

            if i == 0:
                model = abm.opinions(x0, z0, A, C0, D=D)
                domain = model.domain
        
        if param_key == "a":
            items = np.concatenate(
                [
                    items_block, np.array([[theta_ind, theta_inf]] * num_repeat), 
                    np.array([params_sensitivity[param_key]]*num_simulations).reshape(len_param*num_simulations,1), 
                    np.array([[timesteps]] * num_repeat),
                    np.array(list(seeds) * len_param)[:,None]
                ], axis=1)
        elif param_key == "theta_ind":
            items = np.concatenate(
                [
                    items_block, 
                    np.array([params_sensitivity[param_key]] * num_simulations).reshape(len_param*num_simulations,1), 
                    np.array([[theta_inf]] * num_repeat), 
                    np.array([[a]] * num_repeat),
                    np.array([[timesteps]] * num_repeat),
                    np.array(list(seeds) * len_param)[:,None]
                ], axis=1)
        elif param_key == "theta_inf":
            items = np.concatenate(
                [
                    items_block, np.array([[theta_ind]] * num_repeat), 
                    np.array([params_sensitivity[param_key]] * num_simulations).reshape(len_param*num_simulations,1), 
                    np.array([[a]] * num_repeat),
                    np.array([[timesteps]] * num_repeat),
                    np.array(list(seeds) * len_param)[:,None]
                ], axis=1)
        pool = multiprocessing.Pool(processes=len_param)
        results = pool.map(run_model, items) 
        pool.close()

        xs = np.array(results)[:,0,:].reshape((num_simulations, len_param, timesteps+1))
        zs = np.array(results)[:,1,:].reshape((num_simulations, len_param, timesteps+1))

        for param_idx in np.arange(params_len[param_key]):
            for time_point in stop_time_points:
                xs_arr = np.array([xs[i,param_idx,time_point] for i in range(num_simulations)])
                zs_arr = np.array([zs[i,param_idx,time_point] for i in range(num_simulations)])
                
                plt.figure(figsize=(12,8))
                #_, _, _, im1 = plt.hist2d(np.sum(xs_arr, axis=0)[:,0], np.sum(xs_arr, axis=0)[:,1], bins = 20, range = domain)
                #np.sum([plt.hist2d(xs_arr[i][:,0], xs_arr[i][:,1], bins = 20, range = domain)[0] for i in range(num_simulations)])
                hist_matrix = np.mean([plt.hist2d(xs_arr[i][:,0], xs_arr[i][:,1], bins = 20, range = domain)[0] for i in range(num_simulations)], axis=0)
                im1 = plt.imshow(hist_matrix)
                cbar = plt.colorbar(im1)
                plt.savefig(imgpath+"/histogram_x_time_point_{0}_param_{1}_param_idx_{2}.png".format(time_point, param_key, param_idx))
                plt.close()

                plt.figure(figsize=(12,8))
                plt.figure(figsize=(12,8))
                hist_matrix = np.mean([plt.hist2d(zs_arr[i][:,0], zs_arr[i][:,1], bins = 20, range = domain)[0] for i in range(num_simulations)], axis=0)
                im2 = plt.imshow(hist_matrix)
                cbar = plt.colorbar(im2)
                #_, _, _, im2 = plt.hist2d(np.sum(zs_arr, axis=0)[:,0], np.sum(zs_arr, axis=0)[:,1], bins = 20, range = domain)
                plt.savefig(imgpath+"/histogram_z_time_point_{0}_param_{1}_param_idx_{2}.png".format(time_point, param_key, param_idx))
                plt.close()
    stop = time.time()
    print(stop-start)